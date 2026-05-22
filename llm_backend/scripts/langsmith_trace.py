"""LangSmith 全链路追踪脚本

跑 Supervisor + Worker 完整链路并上传 LangSmith trace：
  Case 1: 单意图 product_qa  →  classify → dispatch → product_qa → merge
  Case 2: 多意图 multi        →  classify → decompose → [product_qa ‖ order_qa] → merge
  Case 3: 售后 after_sales    →  classify → dispatch → after_sales(reason 档) → merge

MemoryService / SegmentManager 用轻量 stub 替代（无需 MySQL）。
工具执行器用 mock（无需 Neo4j / Milvus）。
LLM 调用使用真实 API，全程被 LangSmith 自动 trace。

运行方式：
    cd llm_backend
    python scripts/langsmith_trace.py
"""

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings

# pydantic-settings 只把 .env 读入 model，不写 os.environ。
# LangSmith tracer 直接读 os.environ，必须在 langchain 模块实际创建 tracer 前设好。
os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.LANGCHAIN_TRACING_V2 else "false"
os.environ["LANGCHAIN_API_KEY"]    = settings.LANGCHAIN_API_KEY
os.environ["LANGSMITH_API_KEY"]    = settings.LANGCHAIN_API_KEY   # langsmith SDK 用这个
os.environ["LANGCHAIN_PROJECT"]    = settings.LANGCHAIN_PROJECT

from langchain_core.messages import HumanMessage
from app.lg_agent.workers.tools.executors import (
    AskClarificationExecutor,
    EscalateToHumanExecutor,
)
from app.lg_agent.workers.tools.registry import ToolResult, register_tool

# ─── 打印 LangSmith 项目配置 ───────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  LangSmith tracing : {'ON' if settings.LANGCHAIN_TRACING_V2 else 'OFF'}")
print(f"  Project           : {settings.LANGCHAIN_PROJECT}")
print(f"  API key set       : {bool(settings.LANGCHAIN_API_KEY)}")
print(f"{'='*60}\n")

if not settings.LANGCHAIN_TRACING_V2 or not settings.LANGCHAIN_API_KEY:
    print("❌  LangSmith 未启用，请在 .env 中设置：")
    print("    LANGCHAIN_TRACING_V2=True")
    print("    LANGCHAIN_API_KEY=lsv2_...")
    sys.exit(1)


# ─── Mock 工具执行器 ───────────────────────────────────────────────────────────

class _MockExecutor:
    def __init__(self, summary: str, slots: dict | None = None):
        self._summary = summary
        self._slots   = slots or {}

    def invoke(self, args: Dict[str, Any]) -> ToolResult:  # noqa: ARG002
        return ToolResult(
            records=[{"mock": True}],
            summary=self._summary,
            success=True,
            slots=self._slots,
        )


def _install_mock_tools() -> None:
    register_tool("semantic_search",  _MockExecutor(
        "找到 iPhone 16 Pro（¥7999）、华为 Mate 70 Pro（¥6499）",
        slots={"products_mentioned": ["iPhone 16 Pro", "华为 Mate 70 Pro"]},
    ))
    register_tool("compare_products", _MockExecutor(
        "iPhone 16 Pro vs 小米 15 Ultra：A18 Pro vs 骁龙8 Gen4，价差 ¥1500",
    ))
    register_tool("recommend",        _MockExecutor(
        "推荐联想拯救者 Y7000P（¥4999），游戏性能 TOP3",
        slots={"products_mentioned": ["联想拯救者 Y7000P"]},
    ))
    register_tool("track_shipment",   _MockExecutor(
        "订单 #10088：顺丰 SF1234567890，运输中，预计明天到",
        slots={"last_order_id": 10088},
    ))
    register_tool("search_faq",       _MockExecutor(
        "退货政策：7天无理由退货；质量问题30天免运费退换",
    ))
    register_tool("create_ticket",    _MockExecutor("工单 TICKET-MOCK001 已创建"))
    register_tool("ask_clarification", AskClarificationExecutor())
    register_tool("escalate_to_human", EscalateToHumanExecutor())


# ─── MemoryService / SegmentManager stub ──────────────────────────────────────

STUB_SEGMENT_ID = 1
STUB_CTX: Dict[str, Any] = {
    "summary":      "",
    "recent":       [],
    "worker_slots": {},
    "profile":      {},
}


def _patch_memory():
    """返回两个 patch context manager：无副作用地替换 DB 调用"""
    seg_patch = patch(
        "app.services.segment_manager.SegmentManager.get_or_open_segment",
        new_callable=AsyncMock,
        return_value=STUB_SEGMENT_ID,
    )
    ctx_patch = patch(
        "app.services.memory_service.MemoryService.get_classify_context",
        new_callable=AsyncMock,
        return_value=STUB_CTX,
    )
    slot_patch = patch(
        "app.services.memory_service.MemoryService.write_slot",
        new_callable=AsyncMock,
        return_value=None,
    )
    return seg_patch, ctx_patch, slot_patch


# ─── 追踪用例 ─────────────────────────────────────────────────────────────────

TRACE_CASES = [
    {
        "run_name": "trace-single-product",
        "query":    "推荐一款 5000 元以内的游戏本，要求散热好",
        "label":    "Case 1 | 单意图 product_qa（flash classify → tool Worker）",
    },
    {
        "run_name": "trace-multi-parallel",
        "query":    "帮我查一下订单 #10088 到哪了，另外推荐一款小米手机",
        "label":    "Case 2 | 多意图 parallel（product_qa ‖ order_qa 并行 Send API）",
    },
    {
        "run_name": "trace-after-sales",
        "query":    "我买的手机收到就有划痕，已经投诉两次了，必须立刻解决",
        "label":    "Case 3 | 售后 after_sales（reason 档 GPT-5.5，escalate 信号）",
    },
]


# ─── 主函数 ───────────────────────────────────────────────────────────────────

async def run_trace(case: dict, graph) -> str:
    """运行单个 trace，返回 LangSmith run_name（用于后续查 URL）"""
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id":   0,           # stub，MemoryService 已被 patch
        },
        "run_name": case["run_name"],
        "tags":     ["langsmith-trace", "benchmark"],
        "metadata": {"trace_case": case["run_name"]},
    }

    print(f"\n{'─'*60}")
    print(f"  {case['label']}")
    print(f"  Query  : {case['query']}")
    print(f"  RunName: {case['run_name']}")
    print(f"{'─'*60}")

    t0 = time.perf_counter()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=case["query"])]},
        config=config,
    )
    elapsed = time.perf_counter() - t0

    msgs = result.get("messages", [])
    answer = msgs[-1].content if msgs else result.get("final_answer", "")
    print(f"  耗时   : {elapsed:.2f}s")
    print(f"  Intent : {result.get('intent', '?')}")
    workers_used = [r.get("worker_type") for r in result.get("worker_results", [])]
    if workers_used:
        print(f"  Workers: {workers_used}")
    print(f"  Answer : {(answer or '')[:120]}...")
    return case["run_name"]


async def main() -> None:
    _install_mock_tools()

    # 打补丁替换 DB 调用，然后构建图
    patches = _patch_memory()
    for p in patches:
        p.start()

    try:
        # _init_tool_registry 会连 Neo4j/Milvus，用 mock 工具替代
        tool_init_patch = patch(
            "app.lg_agent.lg_builder._init_tool_registry",
            side_effect=_install_mock_tools,  # 用我们的 mock 替换
        )
        tool_init_patch.start()

        # 延迟导入——避免在 patch 生效前触发 DB 初始化
        from app.lg_agent.lg_builder import build_supervisor_graph
        graph = build_supervisor_graph().compile()
        print("✅  Supervisor graph compiled")

        run_names = []
        for case in TRACE_CASES:
            name = await run_trace(case, graph)
            run_names.append(name)
            # 间隔一下，避免 LLM API rate limit
            await asyncio.sleep(2)

    finally:
        tool_init_patch.stop()
        for p in patches:
            p.stop()

    # ─── 查询 LangSmith URL ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  LangSmith Traces")
    print(f"{'='*60}")

    try:
        from langsmith import Client
        ls_client = Client()

        # 等 LangSmith 接收数据
        print("  等待 traces 上传...")
        await asyncio.sleep(5)

        for run_name in run_names:
            try:
                runs = list(ls_client.list_runs(
                    project_name=settings.LANGCHAIN_PROJECT,
                    filter=f'eq(name, "{run_name}")',
                    limit=1,
                ))
                if runs:
                    run = runs[0]
                    url = getattr(run, "url", None) or (
                        f"https://smith.langchain.com/o/{run.session_id}/projects/p/"
                        f"{run.session_id}/runs/{run.id}"
                    )
                    print(f"  {run_name}")
                    print(f"    {url}")
                    print(f"    latency={run.total_tokens} tokens | "
                          f"status={run.status}")
                else:
                    print(f"  {run_name} — 未找到（可能还在上传中）")
            except Exception as e:
                print(f"  {run_name} — 查询失败: {e}")

    except ImportError:
        print("  langsmith SDK 未安装，跳过 URL 查询")
        print(f"  请手动打开 https://smith.langchain.com/")
        print(f"  过滤项目 [{settings.LANGCHAIN_PROJECT}]，按 run_name 查找：")
        for name in run_names:
            print(f"    - {name}")

    print(f"\n{'='*60}")
    print(f"  Dashboard: https://smith.langchain.com/")
    print(f"  Project  : {settings.LANGCHAIN_PROJECT}")
    print(f"  搜索 tags: langsmith-trace")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
