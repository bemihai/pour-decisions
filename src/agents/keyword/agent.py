"""
Keyword-based wine agent for testing tools without LLM rate limits.

This module provides a simple keyword-routing agent that:
- Uses pattern matching to select tools (NO LLM for routing)
- Executes tools locally (database queries, calculations - FREE)
- Uses LLM only for final answer generation (1 LLM call per query)

Compared to the intelligent agent (agent.py):
- Intelligent agent: 2-3 LLM calls per query, handles complex queries
- Keyword agent: 1 LLM call per query, simpler routing, better for testing
"""

from typing import Annotated, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.agents.llm import load_base_model
from src.agents.tools import get_tools
from src.utils import get_config, logger, find_project_root


class KeywordAgentState(TypedDict):
    """State for the keyword-based agent graph."""
    messages: Annotated[list, add_messages]
    query: str
    query_type: str  # cellar, taste, knowledge, pairing, web_search
    tool_results: Dict
    needs_llm: bool


# Keyword patterns for routing
KEYWORD_PATTERNS = {
    "cellar": [
        # Possession keywords
        "my cellar", "my wines", "i have", "i own", "in my collection",
        "my collection", "my inventory", "wines i have",
        # Query keywords
        "show me", "what wines", "which wines", "how many wines",
        "list my", "wines in", "cellar wines",
        # Specific cellar queries
        "burgundy wines", "italian wines", "ready to drink", "aging wines",
        "location", "rack", "bottles"
    ],
    "taste": [
        # Preference keywords
        "my preferences", "my taste", "my favorite", "i like",
        "top rated", "highest rated", "best wines",
        # Profile keywords
        "taste profile", "wine preferences", "what i like",
        "recommend", "recommendation", "suggest", "should i",
        # Rating keywords
        "rated", "rating", "score"
    ],
    "pairing": [
        # Food pairing keywords
        "pair with", "pairing", "goes with", "match with",
        "food", "steak", "fish", "chicken", "cheese",
        "dinner", "meal", "eat", "serve with",
        # Specific foods
        "salmon", "beef", "lamb", "pork", "pasta", "pizza"
    ],
    "knowledge": [
        # Educational keywords
        "what is", "tell me about", "explain", "define",
        "how is", "why is", "difference between",
        # Wine concepts
        "terroir", "appellation", "region", "grape", "varietal",
        "fermentation", "aging", "tannin", "acid",
        # Regions and grapes (when not asking about personal cellar)
        "bordeaux", "burgundy", "tuscany", "napa",
        "cabernet", "pinot noir", "chardonnay"
    ],
    "web_search": [
        # Price keywords
        "price", "cost", "how much", "buy", "purchase", "retail",
        "expensive", "cheap", "value", "worth",
        # Availability keywords
        "where to buy", "available", "in stock", "order", "shop",
        # Review / critic keywords
        "review", "points", "critic",
        "wine advocate", "vinous", "james suckling", "wine spectator",
        "jancis robinson", "decanter",
        # Current information
        "latest", "recent", "current", "news",
        # Market keywords
        "market", "auction", "investment", "resale",
    ]
}


class KeywordWineAgent:
    """
    Keyword-based wine agent for testing without LLM rate limits.

    Uses simple pattern matching to route queries to appropriate tools,
    then uses LLM only once for final answer generation.

    LLM Usage: 1 call per query (final answer only)

    Attributes:
        llm: The language agents for final answer generation
        tools: Dictionary mapping tool names to tool instances
        agent: The compiled LangGraph agent
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        phase: int = 1,
        verbose: bool = False
    ):
        """
        Initialize the keyword-based wine agent.

        Args:
            llm: Language agents instance. If None, loads default from config.
            phase: Implementation phase (1=core tools, 2=all tools). Default 1.
            verbose: If True, shows routing decisions. Default False.
        """
        self.verbose = verbose

        # Load LLM if not provided
        if llm is None:
            config = get_config()
            self.llm = load_base_model(
                config.model.provider,
                config.model.name
            )
            logger.info(f"Loaded default LLM for keyword agent: {config.model.provider}/{config.model.name}")
        else:
            self.llm = llm

        # Get tools and create lookup dictionary
        tool_list = get_tools()
        self.tools = {tool.name: tool for tool in tool_list}
        logger.info(f"Loaded {len(self.tools)} tools for keyword agent")

        # Create agent graph
        self.agent = self._create_agent()
        logger.info("Keyword wine agent initialized successfully")

    def _create_agent(self):
        """Create the keyword-based routing graph."""

        def classify_query(state: KeywordAgentState):
            """Classify query using keyword matching (NO LLM)."""
            query = state["query"].lower()

            # Score each category
            scores = {}
            for category, keywords in KEYWORD_PATTERNS.items():
                score = sum(1 for kw in keywords if kw in query)
                scores[category] = score

            # Determine query type
            max_score = max(scores.values()) if scores else 0

            if max_score == 0:
                query_type = "knowledge"  # Default to knowledge
            else:
                query_type = max(scores.items(), key=lambda x: x[1])[0]

            if self.verbose:
                logger.info(f"Keyword routing: {query_type} (scores: {scores})")

            return {
                "query_type": query_type,
                "needs_llm": True
            }

        def execute_cellar_tools(state: KeywordAgentState):
            """Execute cellar-related tools."""
            query = state["query"].lower()
            results = {}

            # Determine which cellar tool to use
            if any(kw in query for kw in ["statistics", "overview", "how many"]):
                tool = self.tools.get("get_cellar_statistics")
                if tool:
                    results["statistics"] = tool.invoke({})

            elif any(kw in query for kw in ["location", "rack", "shelf"]):
                # Extract location from query
                for word in query.split():
                    if len(word) == 1 and word.isalpha():  # Single letter like "A"
                        tool = self.tools.get("find_wines_by_location")
                        if tool:
                            results["wines"] = tool.invoke({"location": word.upper()})
                        break

            else:
                # General cellar query - extract filters
                filters = {}

                # Extract region
                regions = ["burgundy", "bordeaux", "tuscany", "rioja", "napa", "piedmont"]
                for region in regions:
                    if region in query:
                        filters["region"] = region.capitalize()
                        break

                # Extract wine type
                if "red" in query:
                    filters["wine_type"] = "Red"
                elif "white" in query:
                    filters["wine_type"] = "White"

                # Extract ready to drink
                if "ready" in query or "drink now" in query:
                    filters["ready_to_drink"] = True

                tool = self.tools.get("get_cellar_wines")
                if tool:
                    results["wines"] = tool.invoke(filters)

            return {"tool_results": results}

        def execute_taste_tools(state: KeywordAgentState):
            """Execute taste profile tools."""
            query = state["query"].lower()
            results = {}

            if any(kw in query for kw in ["recommend", "suggestion"]):
                tool = self.tools.get("get_wine_recommendations_from_profile")
                if tool:
                    results["recommendations"] = tool.invoke({"from_cellar_only": True})

            elif any(kw in query for kw in ["top rated", "best", "highest"]):
                tool = self.tools.get("get_top_rated_wines")
                if tool:
                    results["top_wines"] = tool.invoke({"min_rating": 85, "limit": 10})

            else:
                # General taste profile
                tool = self.tools.get("get_user_taste_profile")
                if tool:
                    results["profile"] = tool.invoke({})

            return {"tool_results": results}

        def execute_pairing_tools(state: KeywordAgentState):
            """Execute food pairing tools."""
            query = state["query"].lower()
            results = {}

            # Extract food type
            foods = ["steak", "beef", "salmon", "fish", "chicken", "lamb",
                    "pork", "pasta", "pizza", "cheese"]

            food_found = None
            for food in foods:
                if food in query:
                    food_found = food
                    break

            if food_found:
                tool = self.tools.get("get_food_pairing_wines")
                if tool:
                    results["pairing"] = tool.invoke({
                        "food": food_found,
                        "from_cellar_only": True
                    })
            else:
                # Default pairing
                results["message"] = "Please specify a food type (e.g., steak, salmon, pasta)"

            return {"tool_results": results}

        def execute_knowledge_tools(state: KeywordAgentState):
            """Execute RAG knowledge tools."""
            query = state["query"]
            results = {}

            # Determine which RAG tool to use
            query_lower = query.lower()

            if any(kw in query_lower for kw in ["what is", "define", "meaning of"]):
                # Extract term to define
                tool = self.tools.get("search_wine_term_definition")
                if tool:
                    # Try to extract the term
                    if "what is" in query_lower:
                        term = query_lower.split("what is")[-1].strip().rstrip("?")
                    elif "define" in query_lower:
                        term = query_lower.split("define")[-1].strip().rstrip("?")
                    else:
                        term = query
                    results["knowledge"] = tool.invoke({"term": term})

            elif any(region in query_lower for region in ["burgundy", "bordeaux", "tuscany", "barolo", "rioja", "napa"]):
                # Region-specific query
                tool = self.tools.get("search_wine_region_info")
                if tool:
                    for region in ["burgundy", "bordeaux", "tuscany", "barolo", "rioja", "napa"]:
                        if region in query_lower:
                            results["knowledge"] = tool.invoke({"region": region.capitalize()})
                            break

            elif any(grape in query_lower for grape in ["pinot noir", "cabernet", "chardonnay", "nebbiolo", "sangiovese"]):
                # Grape variety query
                tool = self.tools.get("search_grape_variety_info")
                if tool:
                    for grape in ["pinot noir", "cabernet", "chardonnay", "nebbiolo", "sangiovese"]:
                        if grape in query_lower:
                            results["knowledge"] = tool.invoke({"varietal": grape.title()})
                            break

            else:
                # General knowledge search
                tool = self.tools.get("search_wine_knowledge")
                if tool:
                    results["knowledge"] = tool.invoke({
                        "query": query,
                        "max_results": 5
                    })

            return {"tool_results": results}

        def generate_answer(state: KeywordAgentState):
            """Generate final answer using LLM (ONLY LLM CALL)."""
            from pathlib import Path

            query = state["query"]
            tool_results = state.get("tool_results", {})
            query_type = state.get("query_type", "unknown")

            # Build context from tool results with explicit empty handling
            context_parts = []
            has_data = False

            for key, value in tool_results.items():
                if isinstance(value, dict):
                    if value:  # Non-empty dict
                        context_parts.append(f"{key}: {value}")
                        has_data = True
                    else:
                        context_parts.append(f"{key}: No data returned")
                elif isinstance(value, list):
                    if value:  # Non-empty list
                        context_parts.append(f"{key}: {len(value)} items - {value}")
                        has_data = True
                    else:
                        context_parts.append(f"{key}: No matching items found")
                elif value:  # Non-empty value
                    context_parts.append(f"{key}: {value}")
                    has_data = True
                else:
                    context_parts.append(f"{key}: No data returned")

            if not context_parts:
                context = "No tools were executed for this query. No specific data available from cellar or knowledge base."
            elif not has_data:
                context = "\n".join(context_parts) + "\n\nNote: Tools were executed but returned no matching data."
            else:
                context = "\n".join(context_parts)

            # Load prompt template from file
            prompt_path = Path(find_project_root()) / "src/agents/prompts/keyword_agent_generation_prompt.md"
            try:
                with open(prompt_path, 'r') as f:
                    prompt_template = f.read().strip()
            except FileNotFoundError:
                logger.warning(f"Prompt file not found at {prompt_path}. Using default prompt.")
                prompt_template = "You are a wine expert assistant. Answer based on: {context}"

            # Format the prompt with actual values
            prompt = prompt_template.format(query=query, query_type=query_type, context=context)


            # Call LLM
            response = self.llm.invoke([HumanMessage(content=prompt)])

            return {"messages": [AIMessage(content=response.content)]}

        def route_to_tools(state: KeywordAgentState):
            """Route to appropriate tool execution node."""
            query_type = state.get("query_type", "knowledge")

            if query_type == "cellar":
                return "execute_cellar"
            elif query_type == "taste":
                return "execute_taste"
            elif query_type == "pairing":
                return "execute_pairing"
            elif query_type == "web_search":
                return "execute_web_search"
            else:
                return "execute_knowledge"

        def execute_web_search_tools(state: KeywordAgentState):
            """Execute web search tools for price, review, or general queries."""
            query = state["query"]
            query_lower = query.lower()
            results = {}

            price_kw = ["price", "cost", "how much", "buy", "purchase", "retail", "worth"]
            review_kw = ["review", "score", "points", "critic", "wine advocate", "vinous",
                         "james suckling", "wine spectator", "jancis robinson", "decanter"]

            if any(kw in query_lower for kw in price_kw):
                tool = self.tools.get("search_wine_price")
                if tool:
                    results["web_search"] = tool.invoke({"wine_name": query})
            elif any(kw in query_lower for kw in review_kw):
                tool = self.tools.get("search_wine_reviews")
                if tool:
                    results["web_search"] = tool.invoke({"wine_name": query})
            else:
                tool = self.tools.get("search_web_for_wine")
                if tool:
                    results["web_search"] = tool.invoke({"query": query, "search_type": "general"})

            return {"tool_results": results}

        # Build the graph
        workflow = StateGraph(KeywordAgentState)

        # Add nodes
        workflow.add_node("classify", classify_query)
        workflow.add_node("execute_cellar", execute_cellar_tools)
        workflow.add_node("execute_taste", execute_taste_tools)
        workflow.add_node("execute_pairing", execute_pairing_tools)
        workflow.add_node("execute_knowledge", execute_knowledge_tools)
        workflow.add_node("execute_web_search", execute_web_search_tools)
        workflow.add_node("generate", generate_answer)

        # Set entry point
        workflow.set_entry_point("classify")

        # Add conditional routing from classify to tool execution
        workflow.add_conditional_edges(
            "classify",
            route_to_tools,
            {
                "execute_cellar": "execute_cellar",
                "execute_taste": "execute_taste",
                "execute_pairing": "execute_pairing",
                "execute_knowledge": "execute_knowledge",
                "execute_web_search": "execute_web_search",
            }
        )

        # All tool nodes lead to generate
        workflow.add_edge("execute_cellar", "generate")
        workflow.add_edge("execute_taste", "generate")
        workflow.add_edge("execute_pairing", "generate")
        workflow.add_edge("execute_knowledge", "generate")
        workflow.add_edge("execute_web_search", "generate")

        # Generate leads to END
        workflow.add_edge("generate", END)

        return workflow.compile()

    def invoke(
        self,
        query: str,
        message_history: list[dict] | None = None,
        trace_context: dict[str, str] | None = None,
    ) -> dict:
        """Process a query using keyword-based routing.

        Args:
            query: User's wine-related question.
            message_history: Optional list of prior conversation turns. Each entry
                is a dict with ``"role"`` (``"human"`` or ``"ai"``) and
                ``"content"`` keys.
            trace_context: Optional request trace metadata forwarded to LangGraph.

        Returns:
            Dictionary containing:
            - messages: List of messages.
            - final_answer: The agent's response.
            - query_type: Detected query type.
            - tool_results: Results from tools.
        """
        logger.info(f"Keyword agent processing: {query[:100]}...")

        invoke_payload = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "message_history": message_history or [],
        }
        runnable_config = RunnableConfig(metadata=trace_context) if trace_context else None
        if runnable_config:
            response = self.agent.invoke(invoke_payload, config=runnable_config)
        else:
            response = self.agent.invoke(invoke_payload)

        # Extract final answer
        final_answer = ""
        try:
            for msg in response.get("messages", []):
                if isinstance(msg, AIMessage):
                    content = msg.content
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
            logger.error(f"Error extracting final answer from keyword agent response: {e}", exc_info=True)
            final_answer = "I encountered an error processing the response. Please try again."

        result = {
            "messages": response.get("messages", []),
            "final_answer": final_answer,
            "query_type": response.get("query_type", "unknown"),
            "tool_results": response.get("tool_results", {})
        }

        logger.info(f"Keyword agent completed. Query type: {result['query_type']}")

        return result

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())


def create_keyword_agent(
    phase: int = 1,
    verbose: bool = False,
    llm: Optional[BaseChatModel] = None,
) -> KeywordWineAgent:
    """
    Factory function to create a keyword-based wine agent.

    Args:
        phase: Implementation phase (1=core tools, 2=all tools)
        verbose: If True, shows routing decisions
        llm: Optional pre-loaded LLM to use instead of loading from config.
             Pass the local (Ollama) or cloud (Gemini) model as needed.

    Returns:
        Initialized KeywordWineAgent instance

    Example:
        >>> agent = create_keyword_agent(phase=1, verbose=True)
        >>> result = agent.invoke("What wines do I own?")
        >>> print(result['final_answer'])
    """
    agent = KeywordWineAgent(
        llm=llm,
        phase=phase,
        verbose=verbose
    )

    logger.info(f"Created keyword wine agent (Phase {phase}, verbose={verbose})")

    return agent

