"""端到端验证 topic_changed 自动关段功能

三个场景：
  Case 1 — 同域追问      product_qa slots → 再问产品   → topic_changed=False，不关段
  Case 2 — 跨域切换      product_qa slots → 问订单      → topic_changed=True，关旧段开新段
  Case 3 — 跨段指代消解  新段（空slots）+ prev_slots有产品 → "那款手机多少钱" → rewritten_query 还原具体型号

LLM 调用真实 API，DB 调用全部 mock。
"""
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings

os.environ["LANGCHAIN_TRACING_V2"] = "false"  # 验证脚本不需要 trace
os.environ["LANGCHAIN_API_KEY"]    = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = settings.LANGCHAIN_PROJECT

from langchain_core.messages import HumanMessage

from app.services.llm_factory import LLMFactory
from app.lg_agent.supervisor.nodes import make_classify_node


# ─── 核心执行函数 ──────────────────────────────────────────────────────────────

async def run_classify(
    query: str,
    worker_slots: dict,
    prev_worker_slots: dict,
    segment_id_seq: list[int],   # get_or_open_segment 的返回序列（支持调两次）
) -> dict:
    """运行 classify_intent 节点，返回关键结果供断言"""
    mock_ctx = {
        "summary": "",
        "recent": [],
        "worker_slots": worker_slots,
        "prev_worker_slots": prev_worker_slots,
        "profile": {},
    }

    with (
        patch("app.services.memory_service.MemoryService.get_classify_context",
              new_callable=AsyncMock, return_value=mock_ctx),
        patch("app.services.segment_manager.SegmentManager.end_segment",
              new_callable=AsyncMock) as mock_end,
        patch("app.services.segment_manager.SegmentManager.get_or_open_segment",
              new_callable=AsyncMock, side_effect=segment_id_seq) as mock_open,
    ):
        llm = LLMFactory.create_llm("flash")
        classify_node = make_classify_node(llm)

        state  = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": "test-thread", "user_id": 0}}

        result = await classify_node(state, config=config)

        # 取第一个子任务的 description（SubTask 是 TypedDict，序列化为 dict）
        sub_tasks = result.get("sub_tasks", [])
        description = sub_tasks[0]["description"] if sub_tasks else ""

        return {
            "intent":              result.get("intent"),
            "segment_id":          result.get("segment_id"),
            "description":         description,
            "end_segment_called":  mock_end.called,
            "end_segment_arg":     mock_end.call_args.args[0] if mock_end.called else None,
            "open_call_count":     mock_open.call_count,
        }


# ─── 测试场景定义 ──────────────────────────────────────────────────────────────

CASES = [
    {
        "name": "Case 1 | 同域追问 → topic_changed=False",
        "query": "它有多少种颜色？",
        "worker_slots": {"product_qa": {"product_name": "iPhone 16 Pro", "budget_max": 8000}},
        "prev_worker_slots": {},
        "segment_id_seq": [42],      # 只调一次 get_or_open_segment
        "expect": {
            "end_segment_called": False,
            "segment_id": 42,        # 段没有切换
        },
    },
    {
        "name": "Case 2 | 跨域切换 → topic_changed=True",
        "query": "帮我查一下我的订单状态",
        "worker_slots": {"product_qa": {"product_name": "iPhone 16 Pro", "budget_max": 8000}},
        "prev_worker_slots": {},
        "segment_id_seq": [42, 99],  # 第一次拿旧段42，第二次开新段99
        "expect": {
            "end_segment_called": True,
            "end_segment_arg": 42,   # 关掉的是旧段
            "segment_id": 99,        # 返回新段
        },
    },
    {
        "name": "Case 3 | 跨段指代消解 → rewritten_query 还原产品名",
        "query": "那款手机多少钱？",
        "worker_slots": {},          # 当前段无 slots
        "prev_worker_slots": {"product_qa": {"product_name": "iPhone 16 Pro"}},
        "segment_id_seq": [99],
        "expect": {
            "end_segment_called": False,   # 无当前 slots，不触发 topic_changed
            "contains_in_description": "iPhone",  # rewritten_query 应包含产品名
        },
    },
]


# ─── 主函数 ────────────────────────────────────────────────────────────────────

async def main():
    print(f"\n{'='*62}")
    print("  validate_topic_changed.py")
    print(f"{'='*62}\n")

    passed = 0
    failed = 0

    for case in CASES:
        print(f"{'─'*62}")
        print(f"  {case['name']}")
        print(f"  Query : {case['query']}")

        try:
            result = await run_classify(
                query             = case["query"],
                worker_slots      = case["worker_slots"],
                prev_worker_slots = case["prev_worker_slots"],
                segment_id_seq    = case["segment_id_seq"],
            )

            print(f"  Intent      : {result['intent']}")
            print(f"  segment_id  : {result['segment_id']}")
            print(f"  end_segment : called={result['end_segment_called']}"
                  + (f"  arg={result['end_segment_arg']}" if result['end_segment_called'] else ""))
            print(f"  description : {result['description'][:80]}")

            # 断言
            expect = case["expect"]
            errors = []

            if "end_segment_called" in expect and result["end_segment_called"] != expect["end_segment_called"]:
                errors.append(
                    f"end_segment_called: got {result['end_segment_called']}, "
                    f"want {expect['end_segment_called']}"
                )
            if "end_segment_arg" in expect and result["end_segment_arg"] != expect["end_segment_arg"]:
                errors.append(
                    f"end_segment_arg: got {result['end_segment_arg']}, "
                    f"want {expect['end_segment_arg']}"
                )
            if "segment_id" in expect and result["segment_id"] != expect["segment_id"]:
                errors.append(
                    f"segment_id: got {result['segment_id']}, "
                    f"want {expect['segment_id']}"
                )
            if "contains_in_description" in expect:
                kw = expect["contains_in_description"]
                if kw.lower() not in result["description"].lower():
                    errors.append(
                        f"description should contain '{kw}', got: {result['description'][:80]}"
                    )

            if errors:
                print(f"  ❌ FAIL")
                for e in errors:
                    print(f"       {e}")
                failed += 1
            else:
                print(f"  ✅ PASS")
                passed += 1

        except Exception as exc:
            import traceback
            print(f"  ❌ ERROR: {exc}")
            traceback.print_exc()
            failed += 1

        await asyncio.sleep(1)   # 避免 rate limit

    print(f"\n{'='*62}")
    print(f"  结果: {passed} passed / {failed} failed")
    print(f"{'='*62}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
