"""工具执行器实现 — 薄层，真实逻辑在 DataService"""
import logging
from typing import Any, Dict, List
from .registry import ToolResult, register_tool

_log = logging.getLogger(__name__)


class SemanticSearchExecutor:
    """语义搜索 — 委托 ProductService.search()"""

    def __init__(self, product_service):
        self._svc = product_service

    def invoke(self, args: Dict[str, Any]) -> ToolResult:
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        try:
            records = self._svc.search(query, top_k)
            if not records:
                return ToolResult(
                    records=[], summary="未找到匹配的产品",
                    success=False, error="no_results",
                )
            summary = f"语义搜索 '{query[:30]}' 找到 {len(records)} 个产品"
            return ToolResult(records=records, summary=summary)
        except Exception as e:
            _log.error(f"SemanticSearch failed: {e}")
            return ToolResult(records=[], summary="", error=str(e), success=False)


class CompareProductsExecutor:
    """产品对比 — 委托 ProductService.compare()（批量查询，无 N+1）"""

    def __init__(self, product_service):
        self._svc = product_service

    def invoke(self, args: Dict[str, Any]) -> ToolResult:
        names = args.get("product_names", [])
        try:
            records = self._svc.compare(names)
            if not records:
                return ToolResult(
                    records=[], summary=f"未找到产品 {names} 的信息",
                    success=False, error="no_results",
                )
            summary = f"已获取 {len(names)} 个产品的对比数据"
            return ToolResult(records=records, summary=summary,
                              slots={"products_mentioned": list(names)})
        except Exception as e:
            _log.warning(f"CompareProducts failed: {e}")
            return ToolResult(records=[], summary="", error=str(e), success=False)


class RecommendExecutor:
    """产品推荐 — 委托 ProductService.recommend()（过滤 + 批量补全）"""

    def __init__(self, product_service):
        self._svc = product_service

    def invoke(self, args: Dict[str, Any]) -> ToolResult:
        budget_max = args.get("budget_max")
        budget_min = args.get("budget_min")
        try:
            records = self._svc.recommend(
                scenario=args.get("scenario", ""),
                budget_min=budget_min,
                budget_max=budget_max,
                preferences=args.get("preferences") or [],   # LLM 可能显式传 null
                exclude_names=set(args.get("exclude") or []), # set(None) 会抛 NoneType 不可迭代
                top_k=args.get("top_k") or 3,
            )
            if not records:
                return ToolResult(
                    records=[], summary="未找到符合条件的产品",
                    success=False, error="no_match",
                )
            summary = "推荐 " + ", ".join(
                f"{r['product_name']}(¥{r.get('price', 'N/A')})" for r in records
            )
            slots: dict = {"products_mentioned": [r["product_name"] for r in records if "product_name" in r]}
            if budget_max is not None:
                slots["budget_max"] = budget_max
            if budget_min is not None:
                slots["budget_min"] = budget_min
            return ToolResult(records=records, summary=summary, slots=slots)
        except Exception as e:
            _log.error(f"Recommend failed: {e}")
            return ToolResult(records=[], summary="", error=str(e), success=False)


class TrackShipmentExecutor:
    """物流追踪 — 委托 OrderService.track_shipment()"""

    def __init__(self, order_service):
        self._svc = order_service

    def invoke(self, args: Dict[str, Any]) -> ToolResult:
        order_id = args.get("order_id")
        if not order_id:
            return ToolResult(records=[], error="missing order_id", success=False)
        try:
            record = self._svc.track_shipment(order_id)
            if not record:
                return ToolResult(
                    records=[], summary=f"未找到订单 #{order_id}",
                    success=False, error="not_found",
                )
            shipped = record.get("o.ShippedDate", "未发货")
            summary = f"订单 #{order_id}: 下单 {record.get('o.OrderDate', '未知')}, 发货状态: {shipped}"
            return ToolResult(records=[record], summary=summary,
                              slots={"last_order_id": int(order_id)})
        except Exception as e:
            _log.error(f"TrackShipment failed: {e}")
            return ToolResult(records=[], error=str(e), success=False)


class CreateTicketExecutor:
    """创建工单 — stub，等工单表就绪后接入 OrderService"""

    def __init__(self, db_session_factory=None):
        self._db = db_session_factory

    def invoke(self, args: Dict[str, Any]) -> ToolResult:
        issue_type = args.get("issue_type", "其他")
        description = args.get("description", "")
        priority = args.get("priority", "normal")
        ticket_id = f"TKT-{hash(description) % 100000:05d}"
        summary = f"工单 {ticket_id} 已创建: [{issue_type}] {description[:50]} (优先级: {priority})"
        return ToolResult(
            records=[{"ticket_id": ticket_id, "issue_type": issue_type, "status": "open"}],
            summary=summary,
        )


class AskClarificationExecutor:
    """澄清工具 — 不涉及数据查询"""

    def invoke(self, args: Dict[str, Any]) -> ToolResult:
        question = args.get("question", "")
        missing_field = args.get("missing_field", "")
        options = args.get("options")
        reroute_to = args.get("reroute_to")
        return ToolResult(
            records=[{"question": question, "missing_field": missing_field, "options": options}],
            summary=question,
            control={
                "action": "reroute" if reroute_to else "clarify",
                "question": question,
                "missing_field": missing_field,
                "reroute_to": reroute_to,
                "options": options,
            },
        )


class SearchFAQExecutor:
    """FAQ 搜索 — 委托 PolicyService.search_faq()"""

    def __init__(self, policy_service):
        self._svc = policy_service

    def invoke(self, args: Dict[str, Any]) -> ToolResult:
        keyword = args.get("keyword", "")
        try:
            records = self._svc.search_faq(keyword)
            if not records:
                return ToolResult(
                    records=[], summary=f"未找到关于'{keyword}'的常见问题",
                    success=False, error="no_results",
                )
            summary = f"找到 {len(records)} 条关于'{keyword}'的常见问题"
            return ToolResult(records=records, summary=summary)
        except Exception as e:
            _log.error(f"SearchFAQ failed: {e}")
            return ToolResult(records=[], error=str(e), success=False)


class EscalateToHumanExecutor:
    """转人工工具 — 不涉及数据查询"""

    def __init__(self, db_session_factory=None):
        self._db = db_session_factory

    def invoke(self, args: Dict[str, Any]) -> ToolResult:
        reason = args.get("reason", "")
        summary_text = args.get("summary", "")
        urgency = args.get("urgency", "normal")
        return ToolResult(
            records=[{"reason": reason, "summary": summary_text, "urgency": urgency}],
            summary=f"转人工: {reason[:50]}",
            control={
                "action": "escalate",
                "reason": reason,
                "summary": summary_text,
                "urgency": urgency,
            },
        )
