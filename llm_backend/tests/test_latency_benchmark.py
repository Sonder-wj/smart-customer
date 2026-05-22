"""延迟与成本基准测试

Benchmark A — 并行 vs 顺序 Worker 派发延迟
  使用真实 LLM 调用（mock 工具执行器），计时并行 vs 顺序派发同一批 Worker。
  重复 N 次，报告中位数延迟和节省比例。

Benchmark B — 三档 LLM 路由 vs 统一单档的 Token 成本
  对覆盖各 Worker 类型的请求集，分别记录各档 LLM 实际 token 用量，
  按单价估算成本，对比「三档路由」与「全量 flash」「全量 reason」的费用差。

运行方式：
    pytest tests/test_latency_benchmark.py -v -s -m benchmark
    pytest tests/test_latency_benchmark.py -v -s -m benchmark -k "latency"
    pytest tests/test_latency_benchmark.py -v -s -m benchmark -k "cost"
"""
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import HumanMessage

from app.lg_agent.workers import after_sales, general_chat, order_qa, product_qa
from app.lg_agent.workers.tools.executors import (
    AskClarificationExecutor,
    EscalateToHumanExecutor,
)
from app.lg_agent.workers.tools.registry import TOOL_REGISTRY, ToolResult, register_tool
from app.services.llm_factory import LLMFactory

# ─── Token 追踪 ────────────────────────────────────────────────────────────────

class TokenTracker(BaseCallbackHandler):
    """捕获所有 LLM 调用的 prompt/completion token 用量"""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage") or \
                (response.llm_output or {}).get("usage") or {}
        self.input_tokens  += (usage.get("prompt_tokens")     or
                               usage.get("input_tokens")      or 0)
        self.output_tokens += (usage.get("completion_tokens") or
                               usage.get("output_tokens")     or 0)
        self.calls += 1

    def reset(self) -> None:
        self.input_tokens = self.output_tokens = self.calls = 0

    def cost_yuan(self, price_in_per_1m: float, price_out_per_1m: float) -> float:
        return (self.input_tokens  * price_in_per_1m +
                self.output_tokens * price_out_per_1m) / 1_000_000

    def __repr__(self) -> str:  # noqa: D105
        return (f"TokenTracker(in={self.input_tokens}, "
                f"out={self.output_tokens}, calls={self.calls})")


# ─── Mock 工具执行器 ────────────────────────────────────────────────────────────

class _MockExecutor:
    def __init__(self, summary: str, records: list | None = None,
                 slots: dict | None = None, success: bool = True):
        self._summary  = summary
        self._records  = records or [{"mock": True}]
        self._slots    = slots or {}
        self._success  = success

    def invoke(self, args: Dict[str, Any]) -> ToolResult:  # noqa: ARG002
        return ToolResult(
            records=self._records,
            summary=self._summary,
            success=self._success,
            slots=self._slots,
        )


_MOCK_EXECUTORS: Dict[str, _MockExecutor] = {
    "semantic_search":  _MockExecutor("找到 iPhone16 Pro，价格 ¥7999",
                                      slots={"product_name": "iPhone16 Pro"}),
    "compare_products": _MockExecutor("iPhone16 Pro vs 小米15 Ultra 对比完成"),
    "recommend":        _MockExecutor("推荐联想拯救者 Y7000P（¥4999）",
                                      slots={"products_mentioned": ["联想拯救者 Y7000P"]}),
    "track_shipment":   _MockExecutor("订单 #10088 运输中，顺丰 SF1234567890",
                                      slots={"last_order_id": 10088}),
    "search_faq":       _MockExecutor("7 天无理由退货，质量问题免运费"),
    "create_ticket":    _MockExecutor("工单 TICKET-MOCK001 已创建"),
}


def _install_mocks() -> Dict[str, Any]:
    """注入 mock 执行器，返回原始注册表快照以便还原"""
    original = dict(TOOL_REGISTRY)
    # 无条件注册所有 mock（无论 TOOL_REGISTRY 是否已有该 key）
    for name, executor in _MOCK_EXECUTORS.items():
        register_tool(name, executor)
    register_tool("ask_clarification", AskClarificationExecutor())
    register_tool("escalate_to_human", EscalateToHumanExecutor())
    return original


def _restore_registry(original: Dict[str, Any]) -> None:
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


# ─── Worker 构建辅助 ────────────────────────────────────────────────────────────

_WORKER_META = {
    "product_qa":   (product_qa,   "tool"),
    "order_qa":     (order_qa,     "tool"),
    "after_sales":  (after_sales,  "reason"),
    "general_chat": (general_chat, "flash"),
}


def _build(worker_type: str, tier_override: str | None = None):
    module, default_tier = _WORKER_META[worker_type]
    tier = tier_override or default_tier
    return module.build(LLMFactory.create_llm(tier))


async def _run_worker(worker, query: str,
                      tracker: TokenTracker | None = None) -> dict:
    state = {"messages": [HumanMessage(content=query)]}
    cfg   = {"callbacks": [tracker]} if tracker else {}
    result = await worker.ainvoke(state, config=cfg)
    return result.get("worker_results", [{}])[0]


# ─── 定价表（¥ / 百万 token） ────────────────────────────────────────────────

PRICING = {
    "flash":  {"in": 0.14,  "out": 0.28},   # DeepSeek-chat（官方价）
    "tool":   {"in": 0.50,  "out": 1.50},   # GPT-5.4-mini via aihubmix（估算）
    "reason": {"in": 5.00,  "out": 15.00},  # GPT-5.5 via aihubmix（估算）
}


# ─── 测试用例 ──────────────────────────────────────────────────────────────────

@dataclass
class BenchCase:
    case_id:       str
    worker_type:   str   # 正式路由下使用的 Worker
    query:         str
    actual_tier:   str   # 三档路由下的档位（flash/tool/reason）


BENCH_CASES: List[BenchCase] = [
    BenchCase("rec_laptop",    "product_qa",   "推荐一款 5000 元以内的游戏笔记本",   "tool"),
    BenchCase("compare_phone", "product_qa",   "对比 iPhone16 Pro 和小米15 Ultra",   "tool"),
    BenchCase("order_track",   "order_qa",     "查询订单 #10088 的物流状态",          "tool"),
    BenchCase("faq_return",    "after_sales",  "我买的手机收到有划痕想退货",          "reason"),
    BenchCase("faq_warranty",  "after_sales",  "这款手机的保修期是多久",              "reason"),
    BenchCase("greeting",      "general_chat", "你好请问你能帮我什么",                "flash"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark A — 并行 vs 顺序 Worker 派发延迟
# ══════════════════════════════════════════════════════════════════════════════

LATENCY_REPEATS     = 5    # 重复次数（取中位数）
PARALLEL_PAIR = [
    ("product_qa", "对比 iPhone16 Pro 和小米15 Ultra，哪款更值得买"),
    ("order_qa",   "查询订单 #10088 的物流状态"),
]


@pytest.mark.benchmark
class TestLatencyParallelVsSequential:
    """并行 vs 顺序 Worker 派发延迟对比"""

    @pytest.fixture(scope="class", autouse=True)
    def mock_tools(self):
        original = _install_mocks()
        yield
        _restore_registry(original)

    @pytest.fixture(scope="class")
    def workers(self, mock_tools):  # 确保 mock 在 worker build 前安装
        return {wt: _build(wt) for wt, _ in PARALLEL_PAIR}

    async def _timed_parallel(self, workers: dict) -> float:
        t0 = time.perf_counter()
        await asyncio.gather(*[
            _run_worker(workers[wt], q)
            for wt, q in PARALLEL_PAIR
        ])
        return time.perf_counter() - t0

    async def _timed_sequential(self, workers: dict) -> float:
        t0 = time.perf_counter()
        for wt, q in PARALLEL_PAIR:
            await _run_worker(workers[wt], q)
        return time.perf_counter() - t0

    async def test_latency_benchmark(self, workers):
        par_times:  List[float] = []
        seq_times:  List[float] = []

        for i in range(LATENCY_REPEATS):
            par_times.append(await self._timed_parallel(workers))
            seq_times.append(await self._timed_sequential(workers))

        par_median = statistics.median(par_times)
        seq_median = statistics.median(seq_times)
        saving_pct = (seq_median - par_median) / seq_median * 100
        ratio      = seq_median / par_median

        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  LATENCY BENCHMARK  (N={LATENCY_REPEATS} repeats, 2 Workers)")
        print(f"{'-' * 64}")
        print(f"  {'Run':<6}  {'Parallel (s)':>14}  {'Sequential (s)':>16}")
        print(f"{'-' * 64}")
        for i, (p, s) in enumerate(zip(par_times, seq_times), 1):
            print(f"  {i:<6}  {p:>14.3f}  {s:>16.3f}")
        print(f"{'-' * 64}")
        print(f"  Median  {par_median:>14.3f}  {seq_median:>16.3f}")
        print(f"{'-' * 64}")
        print(f"  Parallel median   : {par_median:.3f}s")
        print(f"  Sequential median : {seq_median:.3f}s")
        print(f"  Saving            : {saving_pct:.1f}%  (sequential / parallel = {ratio:.2f}x)")
        print(f"  理论下限            : parallel ≈ max(T_worker)，本次实测 {ratio:.2f}x 加速")
        print(f"{sep}\n")

        # 并行应该快于顺序（有 LLM 调用，P99 并行必然 ≤ 顺序）
        assert par_median < seq_median, (
            f"并行中位数 {par_median:.3f}s 不应大于顺序 {seq_median:.3f}s"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark B — 三档 LLM 路由 vs 统一单档的 Token 成本
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.benchmark
class TestCostTierVsUniform:
    """三档路由 vs 全量 flash / 全量 reason 的 token 成本对比"""

    @pytest.fixture(scope="class", autouse=True)
    def mock_tools(self):
        original = _install_mocks()
        yield
        _restore_registry(original)

    async def _run_case_with_tier(
        self, case: BenchCase, tier: str
    ) -> TokenTracker:
        tracker = TokenTracker()
        worker  = _build(case.worker_type, tier_override=tier)
        await _run_worker(worker, case.query, tracker=tracker)
        return tracker

    async def test_cost_benchmark(self):
        # 三档路由：用实际分配的档位
        results_3tier:  Dict[str, TokenTracker] = {}
        # 假设全量 flash
        results_flash:  Dict[str, TokenTracker] = {}
        # 假设全量 reason
        results_reason: Dict[str, TokenTracker] = {}

        for case in BENCH_CASES:
            results_3tier[case.case_id]  = await self._run_case_with_tier(case, case.actual_tier)
            results_flash[case.case_id]  = await self._run_case_with_tier(case, "flash")
            results_reason[case.case_id] = await self._run_case_with_tier(case, "reason")

        def total_cost(results: Dict[str, TokenTracker], tier_map: Dict[str, str]) -> float:
            cost = 0.0
            for cid, tracker in results.items():
                tier = tier_map.get(cid, "flash")
                cost += tracker.cost_yuan(PRICING[tier]["in"], PRICING[tier]["out"])
            return cost

        tier_map_3tier  = {c.case_id: c.actual_tier for c in BENCH_CASES}
        tier_map_flash  = {c.case_id: "flash"        for c in BENCH_CASES}
        tier_map_reason = {c.case_id: "reason"       for c in BENCH_CASES}

        cost_3tier  = total_cost(results_3tier,  tier_map_3tier)
        cost_flash  = total_cost(results_flash,  tier_map_flash)
        cost_reason = total_cost(results_reason, tier_map_reason)

        saving_vs_reason = (cost_reason - cost_3tier) / cost_reason * 100 if cost_reason else 0
        overhead_vs_flash = (cost_3tier - cost_flash) / cost_flash * 100  if cost_flash  else 0

        sep = "=" * 80
        print(f"\n{sep}")
        print(f"  COST BENCHMARK  ({len(BENCH_CASES)} 条请求)")
        print(f"  定价：flash ¥0.14+0.28/M  |  tool ¥0.50+1.50/M  |  reason ¥5+15/M  (估算)")
        print(f"{'-' * 80}")
        print(f"  {'Case':<20}  {'Tier':>8}  {'In tok':>8}  {'Out tok':>8}  {'Cost(¥)':>10}")
        print(f"{'-' * 80}")
        for case in BENCH_CASES:
            t = results_3tier[case.case_id]
            tier = case.actual_tier
            c = t.cost_yuan(PRICING[tier]["in"], PRICING[tier]["out"])
            print(f"  {case.case_id:<20}  {tier:>8}  "
                  f"{t.input_tokens:>8}  {t.output_tokens:>8}  {c:>10.6f}")
        print(f"{'-' * 80}")

        # 汇总
        total_in_3tier  = sum(t.input_tokens  for t in results_3tier.values())
        total_out_3tier = sum(t.output_tokens for t in results_3tier.values())
        print(f"  {'[三档路由合计]':<20}           "
              f"  {total_in_3tier:>8}  {total_out_3tier:>8}  {cost_3tier:>10.6f}")

        total_in_flash  = sum(t.input_tokens  for t in results_flash.values())
        total_out_flash = sum(t.output_tokens for t in results_flash.values())
        print(f"  {'[全量 flash]':<20}           "
              f"  {total_in_flash:>8}  {total_out_flash:>8}  {cost_flash:>10.6f}")

        total_in_reason  = sum(t.input_tokens  for t in results_reason.values())
        total_out_reason = sum(t.output_tokens for t in results_reason.values())
        print(f"  {'[全量 reason]':<20}           "
              f"  {total_in_reason:>8}  {total_out_reason:>8}  {cost_reason:>10.6f}")

        print(f"{'-' * 80}")
        print(f"  三档路由 vs 全量 reason：节省 {saving_vs_reason:.1f}%  "
              f"（¥{cost_reason:.6f} → ¥{cost_3tier:.6f}）")
        print(f"  三档路由 vs 全量 flash ：多花 {overhead_vs_flash:.1f}%  "
              f"（¥{cost_flash:.6f} → ¥{cost_3tier:.6f}，换取工具稳定性和推理质量）")
        print(f"{sep}\n")

        # 三档路由成本应介于全量 flash 和全量 reason 之间
        assert cost_flash <= cost_3tier <= cost_reason, (
            f"三档路由成本 {cost_3tier:.6f} 应在 flash({cost_flash:.6f}) "
            f"和 reason({cost_reason:.6f}) 之间"
        )
        # 对比全量 reason，至少节省 20%（reason 覆盖 2/6 案例，其余走 tool/flash）
        assert saving_vs_reason >= 20, (
            f"相比全量 reason 节省 {saving_vs_reason:.1f}%，预期 ≥ 20%"
        )
