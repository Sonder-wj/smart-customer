"""显式记忆服务 — 三层记忆 + Segment 隔离 slots

三层记忆:
  短期 — messages 表 (滑窗 N 条)
  中期 — dialogue_states 表 (slot JSON, 按 (segment_id, worker_type) 隔离)
  长期 — user_profiles 表 (偏好 JSON) + conversations.summary
"""
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy import func as sa_func

from app.core.database import AsyncSessionLocal
from app.core.logger import get_logger
from app.models.conversation import Conversation
from app.models.message import Message

_log = get_logger(service="memory")


class MemoryService:
    """所有记忆读写都走这里 — 不要绕过它直连数据库"""

    # ── 短期:消息历史 ──

    @staticmethod
    async def save_message_pair(thread_id: str, user_text: str, assistant_text: str) -> None:
        """保存一轮对话(用户消息 + 助手回复)。"""
        async with AsyncSessionLocal() as db:
            conv = (await db.execute(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )).scalar_one_or_none()
            if not conv:
                _log.warning(f"save_message_pair: thread {thread_id} 没有对应 Conversation,跳过落库")
                return
            db.add(Message(conversation_id=conv.id, sender="user", content=user_text))
            db.add(Message(conversation_id=conv.id, sender="assistant", content=assistant_text))
            await db.commit()

    @staticmethod
    async def get_recent_messages(thread_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """读取最近 N 条消息(滑窗)。返回时间正序的 dict 列表。"""
        async with AsyncSessionLocal() as db:
            conv = (await db.execute(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )).scalar_one_or_none()
            if not conv:
                return []
            rows = (await db.execute(
                select(Message)
                .where(Message.conversation_id == conv.id)
                .order_by(Message.id.desc())
                .limit(limit)
            )).scalars().all()
            return [
                {"sender": m.sender, "content": m.content, "created_at": m.created_at.isoformat()}
                for m in reversed(rows)
            ]

    @staticmethod
    async def count_messages(thread_id: str) -> int:
        """统计 thread 下的消息总数,用于判断是否触发摘要。"""
        async with AsyncSessionLocal() as db:
            conv = (await db.execute(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )).scalar_one_or_none()
            if not conv:
                return 0
            count = (await db.execute(
                select(sa_func.count()).select_from(Message).where(Message.conversation_id == conv.id)
            )).scalar_one()
            return int(count or 0)

    # ── 中期:segment 隔离 slots ──

    @staticmethod
    async def write_slot(segment_id: int, worker_type: str, slots: Dict[str, Any]) -> None:
        """合并写入 (segment_id, worker_type) 的 slots，同 key 覆盖、新 key 追加。"""
        from app.models.dialogue_state import DialogueState
        if not slots:
            return
        async with AsyncSessionLocal() as db:
            existing = (await db.execute(
                select(DialogueState).where(
                    DialogueState.segment_id == segment_id,
                    DialogueState.worker_type == worker_type,
                )
            )).scalar_one_or_none()
            if existing:
                merged = dict(existing.slots or {})
                merged.update(slots)
                existing.slots = merged
            else:
                db.add(DialogueState(segment_id=segment_id, worker_type=worker_type, slots=slots))
            await db.commit()

    @staticmethod
    async def get_segment_slots(segment_id: int, worker_type: str) -> Dict[str, Any]:
        """读取单个 (segment_id, worker_type) 的 slots。"""
        from app.models.dialogue_state import DialogueState
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                select(DialogueState).where(
                    DialogueState.segment_id == segment_id,
                    DialogueState.worker_type == worker_type,
                )
            )).scalar_one_or_none()
            return dict(row.slots) if row and row.slots else {}

    @staticmethod
    async def get_all_segment_slots(segment_id: int) -> Dict[str, Dict[str, Any]]:
        """读取当前段所有 worker 的 slots，返回 {worker_type: slots_dict}。"""
        from app.models.dialogue_state import DialogueState
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(
                select(DialogueState).where(DialogueState.segment_id == segment_id)
            )).scalars().all()
            return {row.worker_type: dict(row.slots or {}) for row in rows}

    @staticmethod
    async def get_classify_context(
        thread_id: str,
        segment_id: Optional[int],
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """classify_intent 专用：一次调用取齐所有上下文。

        Returns:
            recent:            最近 6 条消息列表
            summary:           对话摘要（超过滑窗的老消息压缩）
            profile:           用户画像 dict
            worker_slots:      {worker_type: slots}，当前段各 worker 隔离的 slots
            prev_worker_slots: {worker_type: slots}，上一个已关闭段的 slots，仅用于跨段指代消解
        """
        recent = await MemoryService.get_recent_messages(thread_id, limit=6)
        summary = await MemoryService.get_summary(thread_id)
        profile: Dict[str, Any] = {}
        if user_id:
            profile = await MemoryService.get_user_profile(int(user_id))
        worker_slots: Dict[str, Dict] = {}
        prev_worker_slots: Dict[str, Dict] = {}
        if segment_id:
            worker_slots = await MemoryService.get_all_segment_slots(segment_id)
            prev_worker_slots = await MemoryService._get_prev_segment_slots(thread_id, segment_id)
        return {
            "recent": recent,
            "summary": summary,
            "profile": profile,
            "worker_slots": worker_slots,
            "prev_worker_slots": prev_worker_slots,
        }

    @staticmethod
    async def _get_prev_segment_slots(
        thread_id: str, current_segment_id: int
    ) -> Dict[str, Dict]:
        """读取最近一个已关闭段的所有 worker slots，供跨段指代消解使用。
        仅读取，不影响写入隔离逻辑。
        """
        from app.models.topic_segment import TopicSegment
        from app.models.dialogue_state import DialogueState
        async with AsyncSessionLocal() as db:
            conv = (await db.execute(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )).scalar_one_or_none()
            if not conv:
                return {}
            prev_seg = (await db.execute(
                select(TopicSegment)
                .where(
                    TopicSegment.conversation_id == conv.id,
                    TopicSegment.ended_at.is_not(None),
                    TopicSegment.id != current_segment_id,
                )
                .order_by(TopicSegment.ended_at.desc())
                .limit(1)
            )).scalar_one_or_none()
            if not prev_seg:
                return {}
            rows = (await db.execute(
                select(DialogueState).where(DialogueState.segment_id == prev_seg.id)
            )).scalars().all()
            return {row.worker_type: dict(row.slots or {}) for row in rows}

    # ── 长期:用户画像 + 对话摘要 ──

    @staticmethod
    async def get_user_profile(user_id: int) -> Dict[str, Any]:
        """读取用户跨会话偏好。不存在则返回空 dict。"""
        from app.models.user_profile import UserProfile
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                select(UserProfile).where(UserProfile.user_id == user_id)
            )).scalar_one_or_none()
            return dict(row.preferences) if row and row.preferences else {}

    @staticmethod
    async def upsert_user_profile(user_id: int, prefs: Dict[str, Any]) -> None:
        """合并写入用户偏好。"""
        from app.models.user_profile import UserProfile
        if not prefs:
            return
        async with AsyncSessionLocal() as db:
            existing = (await db.execute(
                select(UserProfile).where(UserProfile.user_id == user_id)
            )).scalar_one_or_none()
            if existing:
                merged = dict(existing.preferences or {})
                merged.update(prefs)
                existing.preferences = merged
            else:
                db.add(UserProfile(user_id=user_id, preferences=prefs))
            await db.commit()

    @staticmethod
    async def set_summary(thread_id: str, summary: str) -> None:
        """写入对话摘要(覆盖)。"""
        async with AsyncSessionLocal() as db:
            conv = (await db.execute(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )).scalar_one_or_none()
            if not conv:
                return
            conv.summary = summary
            await db.commit()

    @staticmethod
    async def get_summary(thread_id: str) -> Optional[str]:
        """读取对话摘要(超过滑窗的老消息压缩)。"""
        async with AsyncSessionLocal() as db:
            conv = (await db.execute(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )).scalar_one_or_none()
            return conv.summary if conv else None

    @staticmethod
    async def summarize_if_needed(thread_id: str, llm, threshold: int = 10, keep_recent: int = 8) -> None:
        """当消息总数 >= threshold 时,把更早的消息用 LLM 压缩成 summary。"""
        from langchain_core.prompts import ChatPromptTemplate
        count = await MemoryService.count_messages(thread_id)
        if count < threshold:
            return

        async with AsyncSessionLocal() as db:
            conv = (await db.execute(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )).scalar_one_or_none()
            if not conv:
                return
            rows = (await db.execute(
                select(Message)
                .where(Message.conversation_id == conv.id)
                .order_by(Message.id.asc())
            )).scalars().all()

        if len(rows) <= keep_recent:
            return
        old_rows = rows[:-keep_recent]
        text = "\n".join(f"{r.sender}: {r.content[:300]}" for r in old_rows)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是对话摘要助手。把以下客服对话压成 1-2 句话,只保留:"
             "用户咨询的产品/订单关键词、已给出的关键结论。不要逐字复述。"),
            ("human", "{text}"),
        ])
        resp = await (prompt | llm).ainvoke({"text": text})
        summary = getattr(resp, "content", "") or ""
        if summary:
            await MemoryService.set_summary(thread_id, summary.strip())
