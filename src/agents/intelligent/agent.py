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

from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agents.llm import load_base_model
from src.agents.prompt_renderer import render_intelligent_agent_system_prompt
from src.agents.tools import build_tool_registry
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot
from src.utils import get_config, logger


class AgentState(TypedDict):
    """State for the wine agent graph."""
    messages: Annotated[list, add_messages]


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
            verbose: If True, shows agent reasoning steps. Default False.
        """
        self.verbose = verbose
        config = get_config() if llm is None else None
        if tool_registry is None:
            registry_config = config if config is not None else get_config()
            tool_registry = build_tool_registry(registry_config)
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
            available_only=self.tool_registry.registry_enabled,
        )
        self.tools = [definition.tool for definition in self.tool_selection_snapshot.definitions]
        logger.info(f"Loaded {len(self.tools)} tools.")
        self.system_prompt = render_intelligent_agent_system_prompt(self.tool_selection_snapshot)

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
        1. User query -> agent node (tool_llm selects tools)
        2. If tools selected -> execute_tools (run locally, no LLM)
        3. Tool results -> agent node (tool_llm generates final answer)
        4. End

        Hybrid mode architecture (tool_llm != llm):
        1. User query -> agent node (tool_llm selects tools)
        2. If tools selected -> execute_tools (run locally, no LLM)
        3. generate node always runs (llm generates final answer without tool binding)
        4. End

        In both modes there are 2 LLM calls per query (planning + generation).
        In hybrid mode the first call uses the cloud model and the second uses the
        local model, keeping tool-call reliability high while minimising cloud costs.

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
        workflow.add_node("agent", call_model)
        workflow.set_entry_point("agent")
        if self.tools:
            workflow.add_node("tools", ToolNode(self.tools))

        if self.is_hybrid_mode:
            # Hybrid: always finish with the local llm (no tool binding) for the final answer.
            # This applies both when tools were called and when planning chose no tools.
            workflow.add_node("generate", generate_answer)
            if self.tools:
                workflow.add_conditional_edges(
                    "agent",
                    should_continue,
                    {"tools": "tools", "generate": "generate"},
                )
                workflow.add_edge("tools", "generate")
            else:
                workflow.add_edge("agent", "generate")
            workflow.add_edge("generate", END)
            logger.debug("Built hybrid agent graph: agent(tool_llm) -> tools -> generate(llm) -> END")
        else:
            # Standard: loop back to agent after tools for multi-hop or final answer
            if self.tools:
                workflow.add_conditional_edges(
                    "agent",
                    should_continue,
                    {"tools": "tools", END: END},
                )
                workflow.add_edge("tools", "agent")
            else:
                workflow.add_edge("agent", END)
            logger.debug("Built standard agent graph: agent -> tools -> agent -> END")

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

        invoke_payload = {"messages": history_messages + [HumanMessage(content=query)]}
        runnable_config = RunnableConfig(metadata=trace_context) if trace_context else None
        if runnable_config:
            response = self.agent.invoke(invoke_payload, config=runnable_config)
        else:
            response = self.agent.invoke(invoke_payload)

        # Extract final answer from messages
        final_answer = ""
        try:
            if response["messages"]:
                content = response["messages"][-1].content
                # Handle case where content is a list (some LLMs return lists of content blocks)
                if isinstance(content, list):
                    # Extract text from content blocks
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
        except Exception as e:
            logger.error(f"Error extracting final answer from agent response: {e}", exc_info=True)
            final_answer = "I encountered an error processing the response. Please try again."

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
            {"messages": [("user", query)]}, stream_mode="values"
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
    registry = tool_registry if tool_registry is not None else build_tool_registry(get_config())
    agent = WineAgent(
        llm=llm,
        tool_llm=tool_llm,
        tool_registry=registry,
        verbose=verbose,
    )

    mode = "hybrid" if agent.is_hybrid_mode else "standard"
    logger.info(f"Created wine agent (verbose={verbose}, mode={mode})")

    return agent
