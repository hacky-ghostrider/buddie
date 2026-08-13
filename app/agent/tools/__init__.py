"""Agent tools package — RAG, employee HR tools, calculator, mock search."""

from app.agent.tools.base import AgentTool, ToolRegistry
from app.agent.tools.calculator_tool import CalculatorTool, evaluate_arithmetic
from app.agent.tools.employee_tools import build_employee_tools
from app.agent.tools.rag_tool import (
    RAGToolBundle,
    SearchCompanyPolicyTool,
    SearchDocsTool,
    SummarizeTool,
)
from app.agent.tools.search_tool import SearchTool
from app.employees.service import EmployeeService
from app.orchestration.rag_service import RAGService


def build_default_tool_registry(
    rag_service: RAGService,
    *,
    employee_service: EmployeeService | None = None,
) -> ToolRegistry:
    """Create the default tool registry (RAG + employee + utilities).

    Args:
        rag_service: Shared RAG orchestrator (injected once).
        employee_service: Optional structured employee service.

    Returns:
        ``ToolRegistry`` with RAG tools, employee tools, ``calculator``,
        and ``search``.
    """
    rag_bundle = RAGToolBundle(rag_service)
    registry = ToolRegistry(rag_bundle.tools())
    for tool in build_employee_tools(employee_service):
        registry.register(tool)
    registry.register(CalculatorTool())
    registry.register(SearchTool())
    return registry


__all__ = [
    "AgentTool",
    "ToolRegistry",
    "CalculatorTool",
    "evaluate_arithmetic",
    "RAGToolBundle",
    "SearchDocsTool",
    "SummarizeTool",
    "SearchCompanyPolicyTool",
    "SearchTool",
    "build_employee_tools",
    "build_default_tool_registry",
]
