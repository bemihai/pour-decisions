"""
Wine agent implementation using LangGraph with LLM tool selection.

This module creates an agentic wine assistant that can:
- Query user's wine cellar inventory
- Analyze user's taste profile and preferences
- Search wine knowledge base (RAG)
- Provide food and wine pairing recommendations

The agent uses LLM to intelligently select which tools to use based on user queries.
"""

from typing import Annotated, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agents.llm import load_base_model
from src.agents.tools import get_tools
from src.utils import get_config, logger, find_project_root


class AgentState(TypedDict):
    """State for the wine agent graph."""
    messages: Annotated[list, add_messages]


class WineAgent:
    """
    Agentic wine assistant with tool-using capabilities.

    Uses LangGraph's ReAct agent pattern to intelligently select and use tools
    based on user queries. The agent can handle complex multi-step queries by
    automatically chaining tool calls.

    Architecture:
    1. User query received
    2. LLM analyzes query and decides which tool(s) to use (Planning call)
    3. Tools execute locally (database queries, calculations - FREE)
    4. LLM generates natural language response from tool outputs (Generation call)

    LLM Usage: 2-3 calls per query
    - Planning: 1 call
    - Generation: 1 call
    - Tool correction (if needed): 0-1 call

    Attributes:
        llm: The language agents used for reasoning and generation
        tools: List of tools available to the agent
        agent: The compiled LangGraph agent
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        verbose: bool = False
    ):
        """
        Initialize the wine agent.

        Args:
            llm: Language agents instance. If None, loads default from config.
            verbose: If True, shows agent reasoning steps. Default False.
        """
        self.verbose = verbose

        # Load LLM if not provided
        if llm is None:
            config = get_config()
            self.llm = load_base_model(
                config.model.provider,
                config.model.name
            )
            logger.info(f"Loaded default LLM: {config.model.provider}/{config.model.name}")
        else:
            self.llm = llm
            logger.info(f"Using provided LLM: {type(llm).__name__}")

        # Get tools
        self.tools = get_tools()
        logger.info(f"Loaded {len(self.tools)} tools.")

        # Create agent
        self.agent = self._create_agent()
        logger.info("Wine agent initialized successfully")

    def _create_agent(self):
        """
        Create a custom LangGraph workflow with controlled LLM usage.

        Architecture:
        1. User query → call_model (LLM decides which tools to use)
        2. If tools selected → execute_tools (run locally, no LLM)
        3. Tool results → call_model (LLM generates final answer)
        4. End

        Returns:
            Compiled LangGraph workflow

        Notes:
            - Typically 2 LLM calls per query (tool selection + generation)
            - Tools run locally (free, no LLM calls)
            - More control than prebuilt create_react_agent
        """
        # Load system prompt from file
        from pathlib import Path

        prompt_path = Path(find_project_root()) / "src/agents/prompts/intelligent_agent_system_prompt.md"
        try:
            with open(prompt_path, 'r') as f:
                system_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning(f"System prompt file not found at {prompt_path}. Using default prompt.")
            system_prompt = "You are a helpful wine sommelier assistant with access to specialized tools."


        # Bind tools to LLM
        model_with_tools = self.llm.bind_tools(self.tools)

        # Create tool node for executing tools
        tool_node = ToolNode(self.tools)

        def call_model(state: AgentState):
            """Call LLM to either select tools or generate final answer."""
            from langchain_core.messages import SystemMessage

            messages = state["messages"]

            # Inject system prompt at the beginning if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_prompt)] + messages

            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: AgentState):
            """Decide if we should continue to tools or end."""
            messages = state["messages"]
            last_message = messages[-1]

            # If LLM wants to use tools, route to tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            # Otherwise we're done
            return END

        # Build the graph
        # StateGraph takes the TypedDict class (not instance) as the state schema
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )

        # After tools, always go back to agent for final answer
        workflow.add_edge("tools", "agent")

        # Compile the graph
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

        for chunk in self.agent.stream(
            {"messages": [("user", query)]},
            stream_mode="values"
        ):
            yield chunk

    def _extract_tools_used(self, response: dict) -> List[str]:
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

    def get_available_tools(self) -> List[str]:
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

    def add_tools(self, tools: List[BaseTool]):
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
    config_override: Optional[dict] = None
) -> WineAgent:
    """
    Factory function to create a wine agent instance.

    Convenience function that handles configuration and initialization.

    Args:
        verbose: If True, agent shows reasoning steps. Default False.
        config_override: Optional dict to override default config settings.
                        Example: {"agents": {"name": "gemini-2.0-flash-exp"}}

    Returns:
        Initialized WineAgent instance ready to process queries

    Example:
        >>> agent = create_wine_agent(verbose=True)
        >>> result = agent.invoke("What wines do I own?")

    Notes:
        - Phase 1: 5 core tools (cellar, taste, RAG, pairing)
        - Phase 2: 17 tools (all capabilities)
        - Verbose mode useful for debugging and understanding agent behavior
    """
    config = get_config()

    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)

    agent = WineAgent(
        llm=None,
        verbose=verbose
    )

    logger.info(f"Created wine agent (verbose={verbose})")

    return agent

