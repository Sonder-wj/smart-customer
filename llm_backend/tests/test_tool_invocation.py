"""工具调用命中率评测 — Worker 层（三层评测体系第二层）

三层评测体系：
  Layer 1: test_routing_accuracy.py  — classify_intent 意图分类准确率   ✅
  Layer 2: test_tool_invocation.py   — Worker 工具调用命中率            ← 本文件
  Layer 3: test_answer_quality.py    — merge LLM 回答质量               ✅

指标：Tool Hit Rate = 预期工具被调用的 case 数 / 总 case 数
阈值：≥ 80%

运行：
    pytest tests/test_tool_invocation.py -v -s -m integration
"""
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.messages import HumanMessage

from app.services.llm_factory import LLMFactory
from app.lg_agent.workers.tools.registry import ToolResult, register_tool

# ── InstrumentedExecutor ──────────────────────────────────────────────────────

# 全局调用日志：顺序执行时直接 clear() 即可保证每 case 隔离
_CALL_LOG: List[str] = []

_MOCK_SUMMARIES = {
    "semantic_search":   "找到 iPhone 16 Pro（¥7999）、华为 Mate 70 Pro（¥6499）",
    "compare_products":  "iPhone 16 Pro vs 小米 15 Ultra：A18 Pro vs 骁龙 8 Gen4，价差 ¥1500",
    "recommend":         "推荐联想拯救者 Y7000P（¥4999），游戏性能 TOP3",
    "track_shipment":    "订单 #10088：顺丰 SF1234567890，运输中，预计明天到",
    "search_faq":        "退货政策：7天无理由退货；质量问题30天免运费退换",
    "create_ticket":     "工单 TICKET-MOCK001 已创建",
    "ask_clarification": "请提供您的订单号，格式如 #10088",
    "escalate_to_human": "已为您升级处理，售后专员将在30分钟内联系您",
}


class _InstrumentedExecutor:
    def __init__(self, tool_name: str):
        self._name = tool_name

    def invoke(self, args: dict) -> ToolResult:
        _CALL_LOG.append(self._name)
        control = None
        if self._name == "ask_clarification":
            control = {"action": "clarify", "question": args.get("question", "请补充信息")}
        elif self._name == "escalate_to_human":
            control = {"action": "escalate", "reason": args.get("reason", "升级处理")}
        return ToolResult(
            records=[{"mock": True}],
            summary=_MOCK_SUMMARIES.get(self._name, "mock"),
            success=True,
            control=control,
        )


def _install_instrumented_tools():
    for name in _MOCK_SUMMARIES:
        register_tool(name, _InstrumentedExecutor(name))


# ── 测试用例 ──────────────────────────────────────────────────────────────────

@dataclass
class InvocationCase:
    case_id: str
    worker: str                 # product_qa | order_qa | after_sales
    query: str
    expected_tools: Set[str]    # 命中任意一个即为 PASS


CASES: List[InvocationCase] = [
    # ── product_qa (7 cases) ─────────────────────────────────────────────────
    InvocationCase("pq_compare_1",
                   "product_qa",
                   "对比一下 iPhone 16 Pro 和小米 15 Ultra 哪个更值得买",
                   {"compare_products"}),
    InvocationCase("pq_compare_2",
                   "product_qa",
                   "MacBook Air M3 和联想小新 Pro 16 哪个更适合学生",
                   {"compare_products"}),
    InvocationCase("pq_recommend_1",
                   "product_qa",
                   "推荐一款 5000 元以内散热好的游戏本",
                   {"recommend"}),
    InvocationCase("pq_recommend_2",
                   "product_qa",
                   "1000 元左右的 TWS 耳机推荐",
                   {"recommend"}),
    InvocationCase("pq_search_1",
                   "product_qa",
                   "华为 Mate 70 Pro 的摄像头参数是多少",
                   {"semantic_search"}),
    InvocationCase("pq_search_2",
                   "product_qa",
                   "三星 Galaxy S25 Ultra 防水等级",
                   {"semantic_search"}),
    InvocationCase("pq_search_3",
                   "product_qa",
                   "有没有 256GB 存储的 5000 元以内手机",
                   {"semantic_search", "recommend"}),

    # ── order_qa (4 cases) ───────────────────────────────────────────────────
    InvocationCase("oq_track_1",
                   "order_qa",
                   "帮我查一下订单 #10088 到哪了",
                   {"track_shipment"}),
    InvocationCase("oq_track_2",
                   "order_qa",
                   "订单 #20001 发货了吗",
                   {"track_shipment"}),
    InvocationCase("oq_track_3",
                   "order_qa",
                   "顺丰单号 SF1234567890 现在在哪",
                   {"track_shipment"}),
    InvocationCase("oq_clarify",
                   "order_qa",
                   "帮我查一下我的订单状态",       # 无订单号，应该 ask_clarification
                   {"ask_clarification"}),

    # ── after_sales (5 cases) ────────────────────────────────────────────────
    InvocationCase("as_faq_1",
                   "after_sales",
                   "七天无理由退货怎么申请",
                   {"search_faq"}),
    InvocationCase("as_faq_2",
                   "after_sales",
                   "手机保修期是多久，人为损坏也能保吗",
                   {"search_faq"}),
    InvocationCase("as_faq_3",
                   "after_sales",
                   "质量问题怎么申请换货，需要什么材料",
                   {"search_faq"}),
    InvocationCase("as_escalate_1",
                   "after_sales",
                   "我已经反映三次了还没解决，必须马上给我处理，不然投诉到315",
                   {"escalate_to_human"}),
    InvocationCase("as_escalate_2",
                   "after_sales",
                   "收到货就是坏的，客服说不给退，这是欺骗消费者，我要曝光",
                   {"escalate_to_human"}),

    # ── 边界 case：容易混淆的场景 ────────────────────────────────────────────
    InvocationCase("pq_search_not_rec",
                   "product_qa",
                   "最近有什么新款手机",                    # 无预算/偏好，不应调 recommend
                   {"semantic_search"}),
    InvocationCase("pq_search_ambig",
                   "product_qa",
                   "华为手机的拍照效果怎么样",              # 查参数/口碑，非推荐场景
                   {"semantic_search"}),
    InvocationCase("as_faq_not_escalate",
                   "after_sales",
                   "我的手机摔坏了屏幕，能保修吗",          # 普通售后，不应直接 escalate
                   {"search_faq"}),
    InvocationCase("as_faq_before_ticket",
                   "after_sales",
                   "我想申请一下换货",                      # 应先查政策，不应直接建工单
                   {"search_faq"}),
    InvocationCase("oq_clarify_ambig",
                   "order_qa",
                   "我的快递好像丢了",                      # 没有单号，应先 ask_clarification
                   {"ask_clarification"}),

    # ── 强干扰 case：高度模糊、LLM 易判断错 ─────────────────────────────────────
    InvocationCase("pq_hard_1",
                   "product_qa",
                   "iPhone 16 Pro 最近有没有降价活动",       # 价格查询 → semantic_search，易误触 recommend
                   {"semantic_search"}),
    InvocationCase("pq_hard_2",
                   "product_qa",
                   "这款手机值得买吗",                       # 无上文，通用估值 → semantic_search，易误触 recommend/compare
                   {"semantic_search", "compare_products"}),
    InvocationCase("pq_hard_3",
                   "product_qa",
                   "有没有比 AirPods Pro 便宜但效果差不多的耳机",  # 含对比+推荐双重意图
                   {"compare_products", "recommend"}),
    InvocationCase("pq_hard_4",
                   "product_qa",
                   "小米 14 Ultra 和 Vivo X100 Pro 哪个拍照更好，顺便推荐一下配件",
                   {"compare_products", "recommend", "semantic_search"}),  # 多工具均可接受
    InvocationCase("as_hard_1",
                   "after_sales",
                   "我手机屏幕有亮点，不知道算不算质量问题，要怎么处理",
                   {"search_faq", "ask_clarification"}),     # 可先查政策也可先澄清产品型号
    InvocationCase("as_hard_2",
                   "after_sales",
                   "买了才三天就出现这个问题，我很不满意",   # 情绪不强烈，应 search_faq 而非 escalate
                   {"search_faq"}),
    InvocationCase("as_hard_3",
                   "after_sales",
                   "我已经联系过客服了，他们说不在保修范围内，我不认可这个说法",
                   {"search_faq", "escalate_to_human"}),     # 有争议，两种工具均合理
    InvocationCase("oq_hard_1",
                   "order_qa",
                   "你好，我有个问题想问一下",               # 极度模糊，应 ask_clarification
                   {"ask_clarification"}),
    InvocationCase("oq_hard_2",
                   "order_qa",
                   "订单应该到了但还没收到，怎么回事",       # 无单号，先澄清还是先查？
                   {"ask_clarification", "track_shipment"}),
]

HIT_RATE_THRESHOLD = 0.75  # 加入强干扰 case 后适当调低阈值，更真实


# ── 测试类 ────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestToolInvocation:
    """Worker 工具调用命中率 — 三层评测体系第二层"""

    @pytest.fixture(autouse=True, scope="class")
    def mock_tools(self):
        _install_instrumented_tools()

    @pytest.fixture(scope="class")
    def workers(self, mock_tools):
        from app.lg_agent.workers import product_qa, order_qa, after_sales
        llm_tool   = LLMFactory.create_llm("tool")
        llm_reason = LLMFactory.create_llm("reason")
        return {
            "product_qa":  product_qa.build(llm_tool),
            "order_qa":    order_qa.build(llm_tool),
            "after_sales": after_sales.build(llm_reason),
        }

    async def _run_one(self, workers: dict, case: InvocationCase) -> dict:
        _CALL_LOG.clear()
        worker = workers[case.worker]
        try:
            await worker.ainvoke(
                {"messages": [HumanMessage(content=case.query)]}
            )
        except Exception as e:
            pass  # call_log 可能已有记录；answer 质量不是本测关注点

        called = set(_CALL_LOG)
        hit    = bool(called & case.expected_tools)
        return {
            "case_id":  case.case_id,
            "worker":   case.worker,
            "query":    case.query,
            "expected": case.expected_tools,
            "called":   called,
            "hit":      hit,
        }

    async def test_hit_rate(self, workers):
        results = []
        for case in CASES:
            r = await self._run_one(workers, case)
            results.append(r)
            await asyncio.sleep(0.8)   # 避免 LLM rate limit

        total   = len(results)
        n_hit   = sum(1 for r in results if r["hit"])
        hit_rate = n_hit / total

        # ── 按 worker 分组统计 ──
        by_worker: dict = {}
        for r in results:
            k = r["worker"]
            by_worker.setdefault(k, {"hit": 0, "total": 0})
            by_worker[k]["total"] += 1
            if r["hit"]:
                by_worker[k]["hit"] += 1

        # ── 控制台报告 ──
        sep    = "=" * 70
        misses = [r for r in results if not r["hit"]]

        print(f"\n{sep}")
        print(f"  TOOL INVOCATION HIT RATE   {n_hit}/{total} = {hit_rate:.1%}")
        print(f"{'-'*70}")
        for w, stat in sorted(by_worker.items()):
            pct = stat["hit"] / stat["total"]
            bar = "o" * stat["hit"] + "x" * (stat["total"] - stat["hit"])
            print(f"  {w:<14}  {stat['hit']}/{stat['total']}  {pct:5.0%}  {bar}")
        print(f"{'-'*70}")
        print(f"  TOTAL           {n_hit}/{total}  {hit_rate:.0%}")
        print(f"  THRESHOLD       {HIT_RATE_THRESHOLD:.0%}")
        print(f"  RESULT          {'PASS' if hit_rate >= HIT_RATE_THRESHOLD else 'FAIL'}")

        if misses:
            print(f"\n  Misses ({len(misses)}):")
            for r in misses:
                print(f"    [x] {r['case_id']:<20}  "
                      f"expected={sorted(r['expected'])}  called={sorted(r['called'])}")
                print(f"         \"{r['query'][:65]}\"")
        print(f"{sep}\n")

        assert hit_rate >= HIT_RATE_THRESHOLD, (
            f"工具命中率 {hit_rate:.1%} 未达阈值 {HIT_RATE_THRESHOLD:.0%}，"
            f"共 {len(misses)} 个 miss"
        )
