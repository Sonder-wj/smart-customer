"""Supervisor 节点实现"""
import json
import uuid
from typing import Any, AsyncIterator, Dict, List

from loguru import logger as _log

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command, Send

from app.lg_agent.supervisor.state import SupervisorState, SubTask, WorkerResult
from app.lg_agent.prompts.supervisor.classify import CLASSIFY_SYSTEM_PROMPT, ClassifyOutput
from app.lg_agent.prompts.supervisor.decompose import DECOMPOSE_SYSTEM_PROMPT, DecomposeOutput
from app.lg_agent.prompts.supervisor.merge import MERGE_SYSTEM_PROMPT

MAX_REROUTE = 2  # 重路由上限


def make_classify_node(llm: BaseChatModel):
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFY_SYSTEM_PROMPT),
        ("human", "{query}"),
    ])

    async def classify_intent(state: SupervisorState, *, config) -> Dict[str, Any]:
        from app.services.memory_service import MemoryService
        from app.services.segment_manager import SegmentManager

        msgs = state.get("messages", [])
        raw_query = msgs[-1].content if msgs and hasattr(msgs[-1], "content") else str(msgs[-1])

        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        user_id = (config or {}).get("configurable", {}).get("user_id")

        # 取当前话题段（不存在则新建），并一次性拉取所有上下文
        segment_id = await SegmentManager.get_or_open_segment(thread_id) if thread_id else None
        ctx = await MemoryService.get_classify_context(thread_id, segment_id, user_id)

        context_parts: List[str] = []
        if ctx["summary"]:
            context_parts.append(f"[历史摘要] {ctx['summary']}")
        if ctx["recent"]:
            history_lines = "\n".join(
                f"  {r['sender']}: {r['content'][:100]}" for r in ctx["recent"][:-1]
            )
            if history_lines:
                context_parts.append(f"[近期对话]\n{history_lines}")
        # 当前话题 slots（按 worker 分组）
        for worker_type, wslots in ctx["worker_slots"].items():
            if wslots:
                label = "用户偏好" if worker_type == "_profile" else f"当前话题:{worker_type}记录"
                context_parts.append(f"[{label}] {json.dumps(wslots, ensure_ascii=False)}")
        # 上个话题 slots（仅用于指代消解，不参与 topic_changed 判断）
        for worker_type, wslots in ctx.get("prev_worker_slots", {}).items():
            if wslots and worker_type != "_profile":
                context_parts.append(
                    f"[上个话题:{worker_type}记录] {json.dumps(wslots, ensure_ascii=False)}"
                )
        if ctx["profile"]:
            context_parts.append(f"[用户画像] {json.dumps(ctx['profile'], ensure_ascii=False)}")

        if context_parts:
            context_block = "\n".join(context_parts)
            enriched_query = f"{context_block}\n\n[当前消息] {raw_query}"
        else:
            enriched_query = raw_query

        result = await (classify_prompt | llm.with_structured_output(ClassifyOutput)).ainvoke(
            {"query": enriched_query}
        )

        # 话题切换：关旧段、开新段，新段无旧 slots，防止跨话题污染
        if result.topic_changed and segment_id and thread_id:
            await SegmentManager.end_segment(segment_id)
            segment_id = await SegmentManager.get_or_open_segment(thread_id)
            _log.info(f"Topic changed → closed old segment, opened new segment {segment_id}")

        # 使用 rewritten_query（若非空）作为 Worker description；回退到原始消息
        description = result.rewritten_query.strip() if result.rewritten_query.strip() else raw_query
        _log.info(
            f"Classify: intent={result.intent} out_of_scope={result.out_of_scope} "
            f"topic_changed={result.topic_changed} rewritten={description[:60]!r}"
        )

        if result.out_of_scope:
            return {
                "intent": "out_of_scope",
                "next_action": "respond",
                "segment_id": segment_id or 0,
                "final_answer": "很抱歉，您的问题超出了我们的服务范围。我们专注于智能家居产品的咨询服务，如有相关问题欢迎继续提问。",
            }

        if result.intent == "general_chat":
            return {
                "intent": "general_chat",
                "workers": ["general_chat"],
                "sub_tasks": [SubTask(
                    task_id=str(uuid.uuid4()),
                    worker_type="general_chat",
                    description=description,
                    context={},
                    priority=1,
                )],
                "next_action": "dispatch",
                "reroute_count": 0,
                "segment_id": segment_id or 0,
            }

        if result.intent == "multi":
            return {
                "intent": "multi",
                "workers": result.workers,
                "next_action": "decompose",
                "reroute_count": 0,
                "segment_id": segment_id or 0,
            }

        # Single worker
        worker = result.intent
        return {
            "intent": result.intent,
            "workers": [worker],
            "sub_tasks": [SubTask(
                task_id=str(uuid.uuid4()),
                worker_type=worker,
                description=description,
                context={},
                priority=1,
            )],
            "next_action": "dispatch",
            "reroute_count": 0,
            "segment_id": segment_id or 0,
        }

    return classify_intent


def make_decompose_node(llm: BaseChatModel):
    decompose_prompt = ChatPromptTemplate.from_messages([
        ("system", DECOMPOSE_SYSTEM_PROMPT),
        ("human", "用户问题: {question}\n需要的 Worker: {workers}\n\n请拆解为子任务:"),
    ])

    async def decompose_tasks(state: SupervisorState, *, config) -> Dict[str, Any]:
        msgs = state.get("messages", [])
        query = msgs[-1].content if msgs and hasattr(msgs[-1], "content") else str(msgs[-1])
        workers = state.get("workers", [])

        result = await (decompose_prompt | llm.with_structured_output(DecomposeOutput)).ainvoke({
            "question": query,
            "workers": ", ".join(workers),
        })

        sub_tasks = []
        for t in result.sub_tasks:
            sub_tasks.append(SubTask(
                task_id=str(uuid.uuid4()),
                worker_type=t.worker_type,
                description=t.description,
                context=t.context,
                priority=t.priority,
            ))

        _log.info(f"Decompose: {len(sub_tasks)} sub-tasks -> {[t['worker_type'] for t in sub_tasks]}")
        return {"sub_tasks": sub_tasks, "next_action": "dispatch"}

    return decompose_tasks


def dispatch_workers(state: SupervisorState) -> Command:
    """使用 Command(goto=[Send]) 并行分发子任务到各 Worker 子图

    Worker 只接收当前 sub_task 的 description 作为单条 HumanMessage，
    不带历史对话——避免 ReAct agent 看到上轮 AIMessage 复用其内容。
    """
    from langchain_core.messages import HumanMessage

    sub_tasks = state.get("sub_tasks", [])

    sends = []
    for task in sub_tasks:
        worker_state = {
            "messages": [HumanMessage(content=task["description"])],
            "worker_type": task["worker_type"],
            "task": task["description"],
            "context": task.get("context", {}),
            "iteration_count": 0,
            "next_action": "think",
            "tool_to_execute": "",
            "tool_call_history": [],
            "final_answer": "",
            "status": "",
            "clarification_question": "",
        }
        sends.append(Send(task["worker_type"], worker_state))

    _log.info(f"Dispatch: {len(sends)} workers -> {[s.node for s in sends]}")
    return Command(goto=sends)


def make_merge_node(llm: BaseChatModel):
    # 合成只看 Worker 输出 — 不注入对话历史，避免上一轮 worker 的上下文 (如订单号、产品名)
    # 被合成 LLM 误混入本轮回复 (跨 worker 污染)
    merge_prompt = ChatPromptTemplate.from_messages([
        ("system", MERGE_SYSTEM_PROMPT),
        ("human", "Worker 执行结果:\n{worker_results}\n\n请基于以上结果生成用户回复:"),
    ])

    async def merge_results(state: SupervisorState, *, config) -> Dict[str, Any]:
        from app.services.memory_service import MemoryService
        segment_id: int = state.get("segment_id") or 0

        # 持久化本轮各 Worker 回传的 slots — 按 worker_type 隔离写入
        if segment_id:
            for r in state.get("worker_results", []):
                wslots = r.get("slots") or {}
                if wslots:
                    await MemoryService.write_slot(segment_id, r["worker_type"], wslots)

        # 长对话摘要 + 用户画像更新由 langgraph SSE 端在流结束后调度,
        # 不在此处 create_task — 那样会让后台 LLM 调用 (summarize) 的 token 流
        # 被外层 astream_events 捕获,和 merge/respond 的 token 在 SSE 上交错输出。

        results = state.get("worker_results", [])
        reroute_count = state.get("reroute_count", 0)
        summary = " | ".join(
            f"[{r.get('worker_type')}:{r.get('status')}:ctrl={r.get('control_action')}:{(r.get('answer','') or '')[:40]}]"
            for r in results
        )
        _log.info(f"merge_results: {len(results)} results, reroute={reroute_count} :: {summary}")

        if not results:
            return {"final_answer": "抱歉，处理过程中出现了问题，请稍后再试。", "next_action": "respond"}

        # 检查 reroute：Worker 判断分错类，建议转给其他 Worker
        for r in results:
            if r.get("control_action") == "reroute" and r.get("reroute_to"):
                if reroute_count < MAX_REROUTE:
                    target = r["reroute_to"]
                    _log.info(f"Reroute: {r['worker_type']} -> {target} (count={reroute_count + 1})")
                    msgs = state.get("messages", [])
                    query = msgs[-1].content if msgs and hasattr(msgs[-1], "content") else str(msgs[-1])
                    return {
                        "workers": [target],
                        "reroute_count": reroute_count + 1,
                        "sub_tasks": [SubTask(
                            task_id=str(uuid.uuid4()),
                            worker_type=target,
                            description=query,
                            context={"reroute_reason": r.get("answer", "")},
                            priority=1,
                        )],
                        "next_action": "dispatch",
                    }
                else:
                    _log.warning(f"Max reroute exceeded, responding with current results")

        # 检查澄清 — 把澄清问题作为 final_answer，让 respond_node 直接输出
        for r in results:
            if r.get("status") == "clarification_needed":
                clarify_q = r.get("clarification_question", "") or r.get("answer", "")
                return {
                    "final_answer": clarify_q,
                    "pending_clarification": clarify_q,
                    "needs_clarification": True,
                    "next_action": "clarify",
                    "worker_results": [],
                }

        # 单结果快速通道：高置信 + 无兜底话术 + 长度合理 → 直接透传
        if len(results) == 1:
            r = results[0]
            confidence = r.get("confidence", 0.5)
            answer = r.get("answer", "")

            # 规则门禁：检测兜底/失败话术
            _low_quality_markers = [
                "抱歉，我暂时无法",
                "无法处理这个问题",
                "系统开小差",
            ]
            is_fallback = any(m in answer for m in _low_quality_markers)
            is_too_short = len(answer.strip()) < 20

            # 阈值 0.5 (中性置信): 多数对话型回复 (无工具调用 or 工具无明确 success 标记)
            # 落在 0.5,如果不放行 fast path 就会被 merge LLM 二次合成,反而带入历史污染
            if confidence >= 0.5 and not is_fallback and not is_too_short:
                return {"final_answer": answer, "next_action": "respond", "worker_results": []}

            # 不满足快速通道 → 走 LLM 合成做二次加工
            _log.info(
                f"Single result gate failed: confidence={confidence} "
                f"fallback={is_fallback} short={is_too_short} — routing to LLM merge"
            )

        # LLM 合成（多结果 或 不满足快速通道的单结果）：按置信度排序合并
        sorted_results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)

        results_text = "\n---\n".join(
            f"[{r.get('worker_type', 'unknown')}] (置信度: {r.get('confidence', 0):.0%})\n{r.get('answer', '')}"
            for r in sorted_results
        )

        response = await (merge_prompt | llm).ainvoke({
            "worker_results": results_text,
        })

        answer = response.content if hasattr(response, "content") else str(response)
        return {"final_answer": answer, "next_action": "respond", "worker_results": []}

    return merge_results


def make_respond_node(llm: BaseChatModel = None):
    """生成最终回复节点 — 支持流式输出"""

    async def respond_node(state: SupervisorState, *, config) -> Dict[str, Any]:
        answer = state.get("final_answer", "抱歉，我暂时无法回答这个问题。")
        return {"messages": [AIMessage(content=answer)]}

    return respond_node


async def respond_node_stream(state: SupervisorState, *, config) -> AsyncIterator[Dict[str, Any]]:
    """流式版 respond_node — yield AIMessageChunk"""
    answer = state.get("final_answer", "抱歉，我暂时无法回答这个问题。")
    # 按 8 字符块输出，模拟 token 级流式
    for i in range(0, len(answer), 8):
        chunk = answer[i:i + 8]
        yield {"messages": [AIMessageChunk(content=chunk)]}
