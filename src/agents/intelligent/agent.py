"""
Wine agent implementation using LangGraph with LLM tool selection.

This module creates an agentic wine assistant that can:
- Query user's wine cellar inventory
- Analyze user's taste profile and preferences
- Search wine knowledge base (RAG)
- Provide food and wine pairing recommendations

The agent uses LLM to intelligently select which tools to use based on user queries.

Hybrid tool-calling mode:
    Pass a separate ``tool_llm`` (e.g. a reliable cloud model) for the planning /
    tool-selection step while ``llm`` (e.g. a local Gemma 4 model) handles the
    final answer generation. This keeps tool-call reliability high at minimal cloud
    cost: 1 cloud call for planning + 1 local call for generation.
"""

from typing import Annotated, Required, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agents.guardrails import (
    CallBudgetConfig,
    LOOP_DETECTED_EVENT_CODE,
    RELEVANCE_DEFLECTED_EVENT_CODE,
    RELEVANCE_REDIRECT,
    TOOL_CALL_FINGERPRINT_ERROR_CODE,
    LoopDetectionConfig,
    RelevanceConfig,
    SanitizationResult,
    SensitiveOutputSanitizer,
    build_safe_tool_call_wrapper,
    build_fail_soft_message,
    call_budget_triggered,
    detect_duplicate_tool_calls,
    evaluate_relevance,
    load_call_budget_config,
    load_loop_detection_config,
    load_relevance_config,
    prepare_model_call,
    relevance_was_deflected,
)
from src.agents.llm import load_base_model
from src.agents.prompt_renderer import render_intelligent_agent_system_prompt
from src.agents.tools import build_tool_registry
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot
from src.utils import get_config, logger


class AgentState(TypedDict, total=False):
    """State for the wine agent graph."""

    messages: Required[Annotated[list, add_messages]]
    llm_call_count: int
    tool_call_history: list[dict[str, str]]
    guardrail_events: list[dict[str, str | int | bool]]


def _finalize_agent_answer(
    response: dict,
    sanitizer: SensitiveOutputSanitizer,
) -> SanitizationResult:
    """Extract and sanitize the final intelligent-agent answer in one shared path."""
    final_answer = ""
    try:
        if response["messages"]:
            content = response["messages"][-1].content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                    else:
                        text_parts.append(str(item))
                final_answer = " ".join(text_parts) if text_parts else ""
            elif isinstance(content, str):
                final_answer = content
            else:
                final_answer = str(content)
    except Exception as exc:
        logger.error("Error extracting final answer from agent response: %s", exc, exc_info=True)
        final_answer = "I encountered an error processing the response. Please try again."
    return sanitizer.sanitize(final_answer)


class WineAgent:
    """
    Agentic wine assistant with tool-using capabilities.

    Uses LangGraph's ReAct agent pattern to intelligently select and use tools
    based on user queries. The agent can handle complex multi-step queries by
    automatically chaining tool calls.

    Architecture (normal mode):
    1. User query received
    2. LLM analyzes query and decides which tool(s) to use (Planning call)
    3. Tools execute locally (database queries, calculations - FREE)
    4. LLM generates natural language response from tool outputs (Generation call)

    Architecture (hybrid mode, tool_llm != llm):
    1. User query received
    2. ``tool_llm`` selects and calls tools (1 cloud call -- reliable tool calling)
    3. Tools execute locally (FREE)
    4. ``llm`` generates the final answer from tool outputs (1 local call -- free)

    LLM Usage: 2-3 calls per query (normal) / 1 cloud + 1 local (hybrid)

    Attributes:
        llm: The language model used for final answer generation.
        tool_llm: The language model used for tool selection (may equal ``llm``).
        tool_registry: Registry available to this agent instance.
        tool_selection_snapshot: Immutable tool selection captured at construction.
        system_prompt: Construction-time prompt matching the bound tool snapshot.
        tools: List of tools available to the agent.
        agent: The compiled LangGraph workflow.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tool_llm: BaseChatModel | None = None,
        tool_registry: ToolRegistry | None = None,
        call_budget: CallBudgetConfig | None = None,
        loop_detection: LoopDetectionConfig | None = None,
        relevance: RelevanceConfig | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the wine agent.

        Args:
            llm: Model for final answer generation. Loads default from config if None.
            tool_llm: Model for tool selection / planning. Defaults to ``llm`` when
                None, producing normal (single-model) mode. Pass a separate cloud
                model here to enable hybrid mode: cloud for planning, local for
                generation.
            tool_registry: Explicit tool registry. A fresh registry is constructed
                from application config when omitted.
            call_budget: Validated synchronous call budget. Reviewed defaults are
                used when dependencies are fully injected and this is omitted.
            loop_detection: Validated exact duplicate-call behavior. Reviewed
                defaults are used when fully injected and omitted.
            relevance: Validated conservative relevance behavior. Reviewed
                defaults are used when fully injected and omitted.
            verbose: If True, shows agent reasoning steps. Default False.
        """
        self.verbose = verbose
        config = get_config() if llm is None or tool_registry is None else None
        self.call_budget = call_budget or load_call_budget_config(config)
        self.loop_detection = loop_detection or load_loop_detection_config(config)
        self.relevance = relevance or load_relevance_config(config)
        if tool_registry is None:
            if config is None:
                config = get_config()
            tool_registry = build_tool_registry(config)
        self.tool_registry = tool_registry

        # Load LLM if not provided
        if llm is None:
            self.llm = load_base_model(
                config.model.provider,
                config.model.name
            )
            logger.info(f"Loaded default LLM: {config.model.provider}/{config.model.name}")
        else:
            self.llm = llm
            logger.info(f"Using provided LLM: {type(llm).__name__}")

        # tool_llm is used for the planning / tool-selection call.
        # When None, defaults to self.llm (single-model mode).
        # When set to a different model, enables hybrid mode.
        self.tool_llm = tool_llm if tool_llm is not None else self.llm
        if self.is_hybrid_mode:
            logger.info(
                f"Hybrid tool-calling mode enabled: {type(self.tool_llm).__name__} "
                f"for planning, {type(self.llm).__name__} for generation"
            )

        self.tool_selection_snapshot: ToolSelectionSnapshot = self.tool_registry.select(
            extended=True,
        )
        self.tools = [definition.tool for definition in self.tool_selection_snapshot.definitions]
        logger.info(f"Loaded {len(self.tools)} tools.")
        self.system_prompt = render_intelligent_agent_system_prompt(self.tool_selection_snapshot)
        self.output_sanitizer = SensitiveOutputSanitizer()

        # Create agent
        self.agent = self._create_agent()
        logger.info("Wine agent initialized successfully")

    @property
    def is_hybrid_mode(self) -> bool:
        """Return True when tool_llm and llm are different instances (hybrid mode)."""
        return self.tool_llm is not self.llm

    def _create_agent(self):
        """
        Create a custom LangGraph workflow with controlled LLM usage.

        Normal mode architecture:
        1. Check deterministic relevance and redirect clear off-topic queries
        2. Reserve an attempted call or route to deterministic termination
        3. User query -> agent node (tool_llm selects tools)
        4. If tools selected -> check exact loops -> execute tools locally
        5. Tool results -> reserve another call -> agent node -> end

        Hybrid mode architecture (tool_llm != llm):
        1. Check deterministic relevance and redirect clear off-topic queries
        2. Reserve an attempted call -> agent node (tool_llm selects tools)
        3. If tools selected -> check exact loops -> execute tools locally
        4. Reserve another attempted call or route to deterministic termination
        5. Generate the final answer without tool binding -> end

        Direct standard answers use one model call; tool-backed standard answers
        use additional planning/finalization calls within the configured budget.
        Hybrid mode uses one planning call and one generation call. The first can
        use a cloud model and the second a local model to minimize cloud cost.

        Returns:
            Compiled LangGraph workflow.
        """
        from langchain_core.messages import SystemMessage

        # tool_llm is used for all tool-selection / planning calls.
        # In hybrid mode this is a different (cloud) model from self.llm.
        model_with_tools = self.tool_llm.bind_tools(self.tools)

        def call_model(state: AgentState):
            """Call tool_llm to select tools or generate answer (non-hybrid mode)."""
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        def check_model_budget(state: AgentState):
            """Reserve the next attempted model call or record exhaustion."""
            return prepare_model_call(state, self.call_budget)

        def generate_answer(state: AgentState):
            """Generate final answer using llm without tool binding (hybrid mode only).

            Called after tools have already executed. The llm sees the full conversation
            including tool results and produces the final natural-language answer.
            """
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        def fail_soft_response(state: AgentState):
            """Return a deterministic answer without another model invocation."""
            from langchain_core.messages import AIMessage

            message = build_fail_soft_message(state.get("messages", []))
            sanitized = self.output_sanitizer.sanitize(str(message.content))
            return {"messages": [AIMessage(content=sanitized.text)]}

        def route_after_budget_check(state: AgentState):
            """Route exhausted budgets to deterministic termination."""
            return "fail_soft" if call_budget_triggered(state) else "model"

        def check_tool_loop(state: AgentState):
            """Approve an entire pending tool batch or record a safe violation."""
            history = list(state.get("tool_call_history", []))
            events = list(state.get("guardrail_events", []))
            if not self.loop_detection.enabled:
                return {"tool_call_history": history, "guardrail_events": events}

            messages = state.get("messages", [])
            pending_calls = getattr(messages[-1], "tool_calls", []) if messages else []
            try:
                result = detect_duplicate_tool_calls(pending_calls, history)
            except (TypeError, ValueError):
                events.append({"code": TOOL_CALL_FINGERPRINT_ERROR_CODE})
                return {"tool_call_history": history, "guardrail_events": events}

            if result.event:
                events.append(result.event)
            return {
                "tool_call_history": list(result.history),
                "guardrail_events": events,
            }

        def route_after_loop_check(state: AgentState):
            """Route exact duplicates away from tool execution."""
            events = state.get("guardrail_events", [])
            latest_code = events[-1].get("code") if events else None
            if latest_code in {LOOP_DETECTED_EVENT_CODE, TOOL_CALL_FINGERPRINT_ERROR_CODE}:
                return "fail_soft"
            return "tools"

        def check_relevance(state: AgentState):
            """Record a bounded event for clear off-topic current queries."""
            events = list(state.get("guardrail_events", []))
            messages = state.get("messages", [])
            content = getattr(messages[-1], "content", "") if messages else ""
            query = content if isinstance(content, str) else ""
            decision = evaluate_relevance(query, self.relevance)
            if decision.route == "deflect":
                events.append(
                    {
                        "code": RELEVANCE_DEFLECTED_EVENT_CODE,
                        "route": "deflect",
                    }
                )
            return {"guardrail_events": events}

        def route_after_relevance_check(state: AgentState):
            """Route clear off-topic queries away from all model and tool work."""
            return "redirect" if relevance_was_deflected(state) else "allow"

        def relevance_redirect(_state: AgentState):
            """Return the deterministic wine-scope redirect."""
            from langchain_core.messages import AIMessage

            sanitized = self.output_sanitizer.sanitize(RELEVANCE_REDIRECT)
            return {"messages": [AIMessage(content=sanitized.text)]}

        def should_continue(state: AgentState):
            """Decide if we should continue to tools, generate, or end."""
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            if self.is_hybrid_mode:
                return "generate"
            return END

        workflow = StateGraph(AgentState)
        workflow.add_node("check_relevance", check_relevance)
        workflow.add_node("relevance_redirect", relevance_redirect)
        workflow.add_node("agent", call_model)
        workflow.add_node("check_agent_budget", check_model_budget)
        workflow.add_node("fail_soft", fail_soft_response)
        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            route_after_relevance_check,
            {"allow": "check_agent_budget", "redirect": "relevance_redirect"},
        )
        workflow.add_edge("relevance_redirect", END)
        workflow.add_conditional_edges(
            "check_agent_budget",
            route_after_budget_check,
            {"model": "agent", "fail_soft": "fail_soft"},
        )
        workflow.add_edge("fail_soft", END)
        if self.tools:
            workflow.add_node("check_loop", check_tool_loop)
            workflow.add_node(
                "tools",
                ToolNode(
                    self.tools,
                    handle_tool_errors=False,
                    wrap_tool_call=build_safe_tool_call_wrapper(self.tool_selection_snapshot),
                ),
            )

        if self.is_hybrid_mode:
            # Hybrid: always finish with the local llm (no tool binding) for the final answer.
            # This applies both when tools were called and when planning chose no tools.
            workflow.add_node("generate", generate_answer)
            workflow.add_node("check_generation_budget", check_model_budget)
            workflow.add_conditional_edges(
                "check_generation_budget",
                route_after_budget_check,
                {"model": "generate", "fail_soft": "fail_soft"},
            )
            if self.tools:
                workflow.add_conditional_edges(
                    "agent",
                    should_continue,
                    {"tools": "check_loop", "generate": "check_generation_budget"},
                )
                workflow.add_conditional_edges(
                    "check_loop",
                    route_after_loop_check,
                    {"tools": "tools", "fail_soft": "fail_soft"},
                )
                workflow.add_edge("tools", "check_generation_budget")
            else:
                workflow.add_edge("agent", "check_generation_budget")
            workflow.add_edge("generate", END)
            logger.debug("Built hybrid agent graph with planning and generation budget checks")
        else:
            # Standard: loop back to agent after tools for multi-hop or final answer
            if self.tools:
                workflow.add_conditional_edges(
                    "agent",
                    should_continue,
                    {"tools": "check_loop", END: END},
                )
                workflow.add_conditional_edges(
                    "check_loop",
                    route_after_loop_check,
                    {"tools": "tools", "fail_soft": "fail_soft"},
                )
                workflow.add_edge("tools", "check_agent_budget")
            else:
                workflow.add_edge("agent", END)
            logger.debug("Built standard agent graph with a pre-model budget check")

        return workflow.compile()

    def invoke(
        self,
        query: str,
        message_history: list[dict] | None = None,
        trace_context: dict[str, str] | None = None,
    ) -> dict:
        """Process a wine-related query and return the agent's response.

        The agent will:
        1. Analyze the query (with optional conversation context).
        2. Decide which tool(s) to use.
        3. Execute tools to gather information.
        4. Generate a natural language response.

        Args:
            query: User's wine-related question or request.
            message_history: Optional list of prior conversation turns. Each entry
                is a dict with ``"role"`` (``"human"`` or ``"ai"``) and
                ``"content"`` keys.
            trace_context: Optional request trace metadata forwarded to LangGraph.

        Returns:
            Dictionary containing:
            - messages: List of message objects.
            - final_answer: The agent's final response (string).
            - tools_used: List of tools that were called.
            - intermediate_steps: Agent reasoning steps (if verbose=True).

        Example:
            >>> agent = WineAgent()
            >>> result = agent.invoke("What Burgundy wines do I own?")
            >>> print(result['final_answer'])

        Notes:
            - LLM calls: 2-3 per query (planning + generation).
            - Tool execution: Local and free (database queries, calculations).
            - Automatically handles multi-tool queries.
        """
        from langchain_core.messages import HumanMessage, AIMessage

        logger.info(f"Processing query: {query[:100]}...")

        # Build initial messages from history + current query
        history_messages = []
        for msg in (message_history or []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "human":
                history_messages.append(HumanMessage(content=content))
            elif role == "ai":
                history_messages.append(AIMessage(content=content))

        invoke_payload: AgentState = {
            "messages": history_messages + [HumanMessage(content=query)],
            "llm_call_count": 0,
            "tool_call_history": [],
            "guardrail_events": [],
        }
        runnable_config = RunnableConfig(
            recursion_limit=self.call_budget.max_graph_steps_per_query,
            metadata=trace_context or {},
        )
        response = self.agent.invoke(invoke_payload, config=runnable_config)

        finalization = _finalize_agent_answer(response, self.output_sanitizer)
        final_answer = finalization.text

        # Extract tools used (always, for logging and debugging)
        tools_used = self._extract_tools_used(response)

        # Debug logging to help diagnose tool usage
        if self.verbose or not tools_used:
            message_types = [type(m).__name__ for m in response.get("messages", [])]
            logger.debug(f"Response message types: {message_types}")
            if not tools_used:
                logger.debug("No tools detected in response messages")

        # Build result
        result = {
            "messages": response["messages"],
            "final_answer": final_answer,
            "tools_used": tools_used,
            "llm_call_count": response.get("llm_call_count", 0),
            "tool_call_history": response.get("tool_call_history", []),
            "guardrail_events": response.get("guardrail_events", []),
        }

        # Add intermediate steps only in verbose mode
        if self.verbose:
            result["intermediate_steps"] = response.get("intermediate_steps", [])

        logger.info(f"Query processed successfully. Tools used: {len(tools_used)} - {', '.join(tools_used) if tools_used else 'none'}")

        return result

    def stream(self, query: str):
        """
        Stream the agent's response in real-time.

        Useful for UI applications that want to show agent progress as it works.
        Yields messages as the agent processes the query.

        Args:
            query: User's wine-related question

        Yields:
            Dictionary chunks with agent state updates

        Example:
            >>> agent = WineAgent()
            >>> for chunk in agent.stream("What wines should I drink tonight?"):
            ...     print(chunk)

        Notes:
            - Enables real-time UI updates
            - Shows tool calls as they happen
            - LLM streaming for faster perceived response time
        """
        logger.info(f"Streaming query: {query[:100]}...")

        yield from self.agent.stream(
            {
                "messages": [("user", query)],
                "llm_call_count": 0,
                "tool_call_history": [],
                "guardrail_events": [],
            },
            config=RunnableConfig(recursion_limit=self.call_budget.max_graph_steps_per_query),
            stream_mode="values",
        )

    def _extract_tools_used(self, response: dict) -> list[str]:
        """
        Extract list of tools that were called during agent execution.

        Args:
            response: Agent response dictionary

        Returns:
            List of tool names that were invoked
        """
        tools_used = []

        # Parse messages to find tool calls
        for message in response.get("messages", []):
            # Check for tool_calls attribute (AIMessage with tool calls)
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    # Handle both dict and object formats
                    if isinstance(tool_call, dict):
                        name = tool_call.get("name")
                        if name:
                            tools_used.append(name)
                    elif hasattr(tool_call, "get"):
                        name = tool_call.get("name")
                        if name:
                            tools_used.append(name)
                    else:
                        # Try to access name attribute directly
                        if hasattr(tool_call, "name"):
                            tools_used.append(tool_call.name)

            # Also check for ToolMessage (responses from tools)
            if hasattr(message, "type") and message.type == "tool":
                # ToolMessage indicates a tool was called
                if hasattr(message, "name"):
                    tools_used.append(message.name)

        unique_tools = list(set(tools_used))  # Remove duplicates
        return unique_tools

    def get_available_tools(self) -> list[str]:
        """
        Get list of tools available to this agent.

        Returns:
            List of tool names the agent can use

        Example:
            >>> agent = WineAgent()
            >>> tools = agent.get_available_tools()
            >>> print(f"Agent has {len(tools)} tools available")
        """
        return [tool.name for tool in self.tools]

    def add_tools(self, tools: list[BaseTool]):
        """
        Add additional tools to the agent.

        Useful for extending agent capabilities at runtime.
        Requires recreating the agent graph.

        Args:
            tools: List of additional tool instances to add

        Example:
            >>> agent = WineAgent()
            >>> agent.add_tools([my_custom_tool])

        Notes:
            - Recreates agent graph (takes a few milliseconds)
            - All previous tools are retained
        """
        self.tools.extend(tools)
        self.agent = self._create_agent()
        logger.info(f"Added {len(tools)} tools. Total tools: {len(self.tools)}")


def create_wine_agent(
    verbose: bool = False,
    llm: BaseChatModel | None = None,
    tool_llm: BaseChatModel | None = None,
    tool_registry: ToolRegistry | None = None,
) -> WineAgent:
    """
    Factory function to create a wine agent instance.

    Convenience function that handles configuration and initialization.

    Args:
        verbose: If True, agent shows reasoning steps. Default False.
        llm: Optional pre-loaded LLM for final answer generation.
             Pass the local (Ollama) or cloud (Gemini) model as needed.
        tool_llm: Optional pre-loaded LLM for tool selection / planning.
             When None, defaults to ``llm`` (single-model mode). Pass a
             reliable cloud model here while ``llm`` is a local model to
             enable hybrid mode: cloud for planning, local for generation.
        tool_registry: Optional explicit registry. A fresh configured registry
             is constructed when omitted.
    Returns:
        Initialized WineAgent instance ready to process queries.

    Example:
        >>> agent = create_wine_agent(verbose=True)
        >>> result = agent.invoke("What wines do I own?")

        Hybrid mode:
        >>> from src.agents.llm import load_base_model
        >>> local = load_base_model("ollama", "gemma4:e2b")
        >>> cloud = load_base_model("google", "gemini-2.5-flash")
        >>> agent = create_wine_agent(llm=local, tool_llm=cloud)
    """
    config = get_config()
    registry = tool_registry if tool_registry is not None else build_tool_registry(config)
    agent = WineAgent(
        llm=llm,
        tool_llm=tool_llm,
        tool_registry=registry,
        call_budget=load_call_budget_config(config),
        loop_detection=load_loop_detection_config(config),
        relevance=load_relevance_config(config),
        verbose=verbose,
    )

    mode = "hybrid" if agent.is_hybrid_mode else "standard"
    logger.info(f"Created wine agent (verbose={verbose}, mode={mode})")

    return agent
