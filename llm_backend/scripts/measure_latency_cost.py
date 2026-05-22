"""测量并行延迟节省 + 三档 LLM 成本占比 — 真实数据（非估算）

A. 延迟（并行 vs 串行）
   每个 Worker 单独跑 N_RUNS 次取中位数 → 多意图请求触发 K 个 Worker：
     并行 worker 阶段 ≈ max(t_i)
     串行 worker 阶段 = Σ t_i
     节省 = (Σ − max) / Σ
   supervisor 开销（classify/decompose/merge）在两种情况下相同，对节省比不产生影响。

B. 成本（token 占比，不拼金额）
   全链路跑代表性请求，callback 抓每次 LLM 调用的【档位 + token】：
     - 各档调用数占比
     - 各档 token 占比
     - 低成本档（DeepSeek flash）承载比例 ← 头条指标

依赖：Neo4j + Milvus + MySQL + Ollama 全部在跑。
运行：python scripts/measure_latency_cost.py
"""
import asyncio
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings

os.environ["LANGCHAIN_TRACING_V2"] = "false"  # 测量脚本不追踪

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from app.services.llm_factory import LLMFactory
from app.lg_agent.lg_builder import _init_tool_registry
from app.lg_agent.workers import product_qa, order_qa, after_sales, general_chat

N_RUNS = 3  # 每个 Worker 重复次数，取中位数抵消抖动


# ─── model → tier 映射 ──────────────────────────────────────────────────────────
# 用关键词子串匹配，兼容 API 返回的解析后名称（如 deepseek-chat → deepseek-v4-flash，
# gpt-5.4-mini → gpt-5.4-mini-2026-03-17）。注意 reason 在 tool 之前判断，避免误配。
_MODEL_TIER = [
    ("deepseek",     "flash  (DeepSeek)"),
    ("gpt-5.5",      "reason (GPT-5.5)"),
    ("gpt-5.4-mini", "tool   (GPT-5.4-mini)"),
]


def _tier_of(model_name: str) -> str:
    name = (model_name or "").lower()
    for key, tier in _MODEL_TIER:
        if key and key in name:
            return tier
    return f"other  ({model_name or 'unknown'})"


# ─── token + latency 采集 callback ──────────────────────────────────────────────
class MetricsCollector(BaseCallbackHandler):
    def __init__(self):
        self.calls = []          # {model, tier, prompt_tokens, completion_tokens, latency}
        self._starts = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs):
        dt = time.perf_counter() - self._starts.pop(run_id, time.perf_counter())
        out = response.llm_output or {}
        model = out.get("model_name") or out.get("model") or ""
        usage = out.get("token_usage") or out.get("usage") or {}
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0

        # 回退：从 message.usage_metadata 取
        if (not pt and not ct) and response.generations:
            try:
                msg = response.generations[0][0].message
                um = getattr(msg, "usage_metadata", None) or {}
                pt = um.get("input_tokens", 0) or 0
                ct = um.get("output_tokens", 0) or 0
                if not model:
                    model = (getattr(msg, "response_metadata", {}) or {}).get("model_name", "")
            except (AttributeError, IndexError):
                pass

        self.calls.append({
            "model": model, "tier": _tier_of(model),
            "prompt_tokens": pt, "completion_tokens": ct, "latency": dt,
        })


# ─── A. 延迟测量 ────────────────────────────────────────────────────────────────
async def measure_latency(workers: dict) -> dict:
    """每个 Worker 跑 N_RUNS 次，返回 {worker_type: median_latency}"""
    queries = {
        "product_qa": "推荐一款 5000 元以内散热好的游戏本",
        "order_qa":   "帮我查一下订单 #10088 到哪了",
        "after_sales": "七天无理由退货的流程是什么",
    }
    result = {}
    for wt, q in queries.items():
        samples = []
        for i in range(N_RUNS):
            t0 = time.perf_counter()
            try:
                await workers[wt].ainvoke({"messages": [HumanMessage(content=q)]})
            except Exception as e:
                print(f"    [{wt} run{i+1}] error: {e}")
            samples.append(time.perf_counter() - t0)
            await asyncio.sleep(1)
        med = statistics.median(samples)
        result[wt] = med
        print(f"  {wt:<12} samples={[f'{s:.2f}' for s in samples]}  median={med:.2f}s")
    return result


# ─── B. 成本（token 占比）测量 ──────────────────────────────────────────────────
async def measure_cost(collector: MetricsCollector):
    """全链路跑代表性请求，token 用量已由 collector 采集（通过 graph callbacks）"""
    from app.lg_agent.lg_builder import build_supervisor_graph

    # 全链路需要 MySQL 上下文 — mock 掉记忆读写，只测 LLM 调用分布
    empty_ctx = {"summary": "", "recent": [], "worker_slots": {},
                 "prev_worker_slots": {}, "profile": {}}

    graph = build_supervisor_graph().compile()

    queries = [
        "推荐一款 5000 元以内散热好的游戏本",   # product_qa
        "帮我查一下订单 #10088 到哪了",          # order_qa
        "七天无理由退货怎么申请",                # after_sales
        "你好，在吗",                            # general_chat
        "对比一下 iPhone 16 Pro 和小米 15 Ultra", # product_qa (compare)
    ]

    with (
        patch("app.services.memory_service.MemoryService.get_classify_context",
              new_callable=AsyncMock, return_value=empty_ctx),
        patch("app.services.memory_service.MemoryService.write_slot",
              new_callable=AsyncMock),
        patch("app.services.segment_manager.SegmentManager.get_or_open_segment",
              new_callable=AsyncMock, return_value=1),
        patch("app.services.segment_manager.SegmentManager.end_segment",
              new_callable=AsyncMock),
    ):
        for i, q in enumerate(queries):
            cfg = {"configurable": {"thread_id": f"measure-{i}", "user_id": 0},
                   "callbacks": [collector]}
            try:
                await graph.ainvoke({"messages": [HumanMessage(content=q)]}, config=cfg)
            except Exception as e:
                print(f"    [cost q{i+1}] error: {e}")
            await asyncio.sleep(1)


# ─── 报告 ───────────────────────────────────────────────────────────────────────
def report_latency(lat: dict):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  A. 并行延迟节省（多意图请求）")
    print(f"{'-'*70}")
    for wt, t in lat.items():
        print(f"    {wt:<12} {t:6.2f}s")

    # 2-worker 组合（最常见多意图）：product_qa + order_qa
    pairs = [
        ("product_qa", "order_qa"),
        ("product_qa", "after_sales"),
    ]
    print(f"{'-'*70}")
    for a, b in pairs:
        if a in lat and b in lat:
            seq = lat[a] + lat[b]
            par = max(lat[a], lat[b])
            save = (seq - par) / seq if seq else 0
            print(f"    {a} + {b}:")
            print(f"      串行 = {lat[a]:.2f} + {lat[b]:.2f} = {seq:.2f}s")
            print(f"      并行 = max = {par:.2f}s")
            print(f"      节省 = {save:.1%}")

    # 3-worker
    if all(k in lat for k in ("product_qa", "order_qa", "after_sales")):
        seq3 = sum(lat[k] for k in ("product_qa", "order_qa", "after_sales"))
        par3 = max(lat[k] for k in ("product_qa", "order_qa", "after_sales"))
        print(f"    product_qa + order_qa + after_sales:")
        print(f"      串行 = {seq3:.2f}s  并行 = {par3:.2f}s  节省 = {(seq3-par3)/seq3:.1%}")


def report_cost(collector: MetricsCollector):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  B. 三档 LLM 成本占比（token，非金额）")
    print(f"{'-'*70}")

    by_tier_calls = defaultdict(int)
    by_tier_tokens = defaultdict(int)
    total_calls = 0
    total_tokens = 0
    for c in collector.calls:
        tok = c["prompt_tokens"] + c["completion_tokens"]
        by_tier_calls[c["tier"]] += 1
        by_tier_tokens[c["tier"]] += tok
        total_calls += 1
        total_tokens += tok

    print(f"    {'档位':<24}{'调用数':>8}{'调用占比':>10}{'token':>10}{'token占比':>10}")
    for tier in sorted(by_tier_calls, key=lambda t: -by_tier_tokens[t]):
        cc = by_tier_calls[tier]
        tt = by_tier_tokens[tier]
        print(f"    {tier:<24}{cc:>8}{cc/total_calls if total_calls else 0:>10.1%}"
              f"{tt:>10}{tt/total_tokens if total_tokens else 0:>10.1%}")
    print(f"{'-'*70}")
    print(f"    合计 {total_calls} 次调用 / {total_tokens} tokens")

    flash_calls = sum(v for k, v in by_tier_calls.items() if k.startswith("flash"))
    flash_tokens = sum(v for k, v in by_tier_tokens.items() if k.startswith("flash"))
    if total_calls:
        print(f"\n    低成本档（DeepSeek flash）承载："
              f"{flash_calls/total_calls:.0%} 调用 / {flash_tokens/total_tokens:.0%} token")
    print(f"{sep}\n")


# ─── 主函数 ─────────────────────────────────────────────────────────────────────
async def main():
    print("初始化工具注册表（连接 Neo4j + Milvus）...")
    _init_tool_registry()

    llm_tool   = LLMFactory.create_llm("tool")
    llm_reason = LLMFactory.create_llm("reason")
    llm_flash  = LLMFactory.create_llm("flash")
    workers = {
        "product_qa":  product_qa.build(llm_tool),
        "order_qa":    order_qa.build(llm_tool),
        "after_sales": after_sales.build(llm_reason),
    }

    print(f"\n[A] 延迟测量（每 Worker × {N_RUNS} 次）...")
    lat = await measure_latency(workers)

    print(f"\n[B] 成本占比测量（全链路 5 请求）...")
    collector = MetricsCollector()
    await measure_cost(collector)

    report_latency(lat)
    report_cost(collector)


if __name__ == "__main__":
    asyncio.run(main())
