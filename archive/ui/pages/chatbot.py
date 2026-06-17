"""Chatbot page"""
import re
import streamlit as st

from src.retrieval import analyze_query, boost_by_metadata_match, build_context_from_chunks, build_semantic_context, \
    compress_context, format_sources_for_display
from src.ui.helper.display import CONTENT_STYLE, display_message, make_page_title
from src.ui.resources import load_llm, load_chroma_client, load_retriever, load_intelligent_agent, load_keyword_agent, load_reranker
from src.ui.sidebar import render_sidebar
from src.agents.llm import process_user_prompt
from src.utils import get_config, logger

_WEB_SEARCH_TOOLS = {"search_web_for_wine", "search_wine_price", "search_wine_reviews"}
_SOURCE_RE = re.compile(r"Source:\s*(https?://\S+)")


def _initial_messages() -> list[dict]:
    """Return the archived UI default greeting."""
    return [{"role": "assistant", "answer": "Hello. How can I help you with wine today?"}]


def _extract_web_sources(messages: list) -> list[dict]:
    """Extract web source URLs and titles from ToolMessage content.

    Args:
        messages: LangGraph message list from agent result.

    Returns:
        List of dicts with 'title' and 'url' keys, deduplicated by URL.
    """
    seen: set[str] = set()
    sources: list[dict] = []

    for msg in messages:
        tool_name = getattr(msg, "name", None)
        if tool_name not in _WEB_SEARCH_TOOLS:
            continue
        content = getattr(msg, "content", "") or ""
        urls = _SOURCE_RE.findall(content)
        lines = content.splitlines()
        # Build a map from line index to URL for title lookup
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            # Find the result block: the title line precedes the snippet and source line
            title = url  # fallback
            for i, line in enumerate(lines):
                if url in line and i >= 2:
                    title_line = lines[i - 2].strip()
                    if title_line:
                        title = re.sub(r"^\[\d+] ", "", title_line)
                    break
            sources.append({"title": title, "url": url})

    return sources


def main():
    """Chatbot page - main entry point."""
    # Load cached resources
    model = load_llm()
    chroma_client = load_chroma_client()
    retriever = load_retriever()
    reranker = load_reranker()

    # Load agents (cached)
    intelligent_agent = load_intelligent_agent()
    keyword_agent = load_keyword_agent()

    # Page title and description
    st.set_page_config(page_title="Pour Decisions", page_icon="🍷")
    st.markdown(make_page_title(
        "Pour Decisions",
        "Let the bot choose your bottle 🍷"
    ), unsafe_allow_html=True)

    # Render sidebar
    render_sidebar(retriever=retriever, chroma_client=chroma_client)

    # Initialize the chat messages history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = _initial_messages()

    st.write(CONTENT_STYLE, unsafe_allow_html=True)

    # Display past messages (include web search indicator for historical messages)
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            display_message(message)

    # Process user prompt
    if prompt := st.chat_input("Type your question here"):
        user_message = {"role": "human", "question": prompt}
        display_message(user_message)
        st.session_state.messages.append(user_message)

        # Get selected agent mode
        agent_mode = st.session_state.get("agent_mode", "No Agent (RAG Only)")

        with st.spinner("Thinking...", show_time=True):
            try:
                # Use agents if selected
                if agent_mode == "Intelligent Agent" and intelligent_agent:
                    try:
                        result = intelligent_agent.invoke(prompt)
                        answer = result.get("final_answer", "")

                        # Extract web search sources from tool messages
                        web_sources = _extract_web_sources(result.get("messages", []))
                        st.session_state.last_web_sources = web_sources
                        st.session_state.last_sources = []
                        st.session_state.last_retrieved_docs = []

                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        logger.error(f"Error using intelligent agent ({error_type}): {error_msg}", exc_info=True)

                        # Provide more specific error message based on error type
                        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                            answer = "The AI service quota has been exceeded. The free tier allows 20 requests per day. Please try again later or switch to 'No Agent (RAG Only)' mode."
                        elif "ChatGoogleGenerativeAI" in error_type or "APIError" in error_type:
                            answer = f"There was an issue with the AI service. Please try again later or switch to 'No Agent (RAG Only)' mode. (Error: {error_type})"
                        elif "AttributeError" in error_type:
                            answer = f"I encountered a data formatting error. Please try rephrasing your question or switch to a different agent mode. (Error: {error_type})"
                        elif "KeyError" in error_type:
                            answer = f"I encountered a missing data error. Please try rephrasing your question or switch to a different agent mode. (Error: {error_type})"
                        else:
                            answer = f"I apologize, but I encountered an error processing your request with the intelligent agent. Please try again or switch to a different agent mode. (Error: {error_type})"

                        # Clear sources
                        st.session_state.last_sources = []
                        st.session_state.last_web_sources = []
                        st.session_state.last_retrieved_docs = []

                elif agent_mode == "Keyword Agent" and keyword_agent:
                    try:
                        result = keyword_agent.invoke(prompt)
                        answer = result.get("final_answer", "")

                        # Extract web sources from keyword agent tool_results
                        tool_results = result.get("tool_results", {})
                        web_text = tool_results.get("web_search", "")
                        web_sources = []
                        if web_text:
                            seen: set[str] = set()
                            lines = web_text.splitlines()
                            for i, line in enumerate(lines):
                                m = _SOURCE_RE.search(line)
                                if m:
                                    url = m.group(1)
                                    if url not in seen:
                                        seen.add(url)
                                        title = url
                                        if i >= 2:
                                            title_line = lines[i - 2].strip()
                                            if title_line:
                                                title = re.sub(r"^\[\d+] ", "", title_line)
                                        web_sources.append({"title": title, "url": url})
                        st.session_state.last_web_sources = web_sources
                        st.session_state.last_sources = []
                        st.session_state.last_retrieved_docs = []

                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        logger.error(f"Error using keyword agent ({error_type}): {error_msg}", exc_info=True)

                        # Provide more specific error message based on error type
                        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                            answer = "The AI service quota has been exceeded. The free tier allows 20 requests per day. Please try again later or switch to 'No Agent (RAG Only)' mode."
                        elif "ChatGoogleGenerativeAI" in error_type or "APIError" in error_type:
                            answer = f"There was an issue with the AI service. Please try again later or switch to 'No Agent (RAG Only)' mode. (Error: {error_type})"
                        else:
                            answer = f"I apologize, but I encountered an error processing your request with the keyword agent. Please try again or switch to a different agent mode. (Error: {error_type})"

                        # Clear sources
                        st.session_state.last_sources = []
                        st.session_state.last_web_sources = []
                        st.session_state.last_retrieved_docs = []

                else:
                    # No Agent mode - use traditional RAG
                    message_history = st.session_state.messages.copy()
                    context = ""

                    cfg = get_config()

                    # Check if RAG is enabled and retrieval is available
                    if st.session_state.get("enable_rag", True) and retriever is not None:
                        try:
                            # Get user-selected number of results or use default
                            n_results = st.session_state.get("n_results", cfg.chroma.retrieval.n_results)

                            # Retrieve more docs if reranking is enabled (reranker will filter down)
                            retrieve_count = n_results * 2 if reranker else n_results

                            # Analyze query for metadata-based filtering/boosting
                            query_analysis = analyze_query(prompt)

                            # Retrieve relevant documents from ChromaDB (or hybrid search)
                            retrieved_docs = retriever.retrieve(prompt, n_results=retrieve_count)

                            # Boost results that match query entities in metadata
                            enable_metadata_boost = getattr(cfg.chroma.retrieval, 'enable_metadata_boost', True)
                            if enable_metadata_boost and query_analysis.has_filters and retrieved_docs:
                                boost_factor = getattr(cfg.chroma.retrieval, 'metadata_boost_factor', 0.1)
                                retrieved_docs = boost_by_metadata_match(
                                    retrieved_docs, query_analysis, boost_factor=boost_factor
                                )
                                logger.debug(f"Applied metadata boosting for: {query_analysis.get_boost_terms()}")

                            # Apply reranking if enabled
                            if reranker and retrieved_docs:
                                rerank_top_k = getattr(cfg.chroma.retrieval, 'rerank_top_k', n_results)
                                retrieved_docs = reranker.rerank(prompt, retrieved_docs, top_k=rerank_top_k)
                                logger.debug(f"Reranked to top {rerank_top_k} documents")

                            # Expand to parent context if small-to-big is enabled
                            enable_small_to_big = getattr(cfg.chroma.chunking, 'enable_small_to_big', False)
                            if enable_small_to_big and retrieved_docs:
                                from src.chroma.hierarchical_chunks import expand_to_parent_context
                                retrieved_docs = expand_to_parent_context(retrieved_docs)
                                logger.debug("Expanded to parent context (small-to-big)")

                            # Build context from retrieved chunks with optional deduplication
                            if cfg.chroma.retrieval.use_deduplication:
                                context = build_semantic_context(
                                    retrieved_docs,
                                    similarity_threshold=cfg.chroma.retrieval.deduplication_threshold,
                                    include_metadata=True,
                                    embedding_model=cfg.chroma.settings.embedder
                                )
                            else:
                                context = build_context_from_chunks(
                                    retrieved_docs,
                                    include_metadata=True,
                                    include_similarity=False,
                                    max_chunks=None
                                )

                            # Apply context compression if enabled
                            enable_compression = getattr(cfg.chroma.retrieval, 'enable_compression', False)
                            if enable_compression and context:
                                max_chars = getattr(cfg.chroma.retrieval, 'compression_max_chars', 8000)
                                context = compress_context(context, max_chars=max_chars)

                            # Store retrieved docs in session state for sidebar display
                            if retrieved_docs:
                                st.session_state.last_sources = format_sources_for_display(retrieved_docs)
                                st.session_state.last_retrieved_docs = retrieved_docs
                            else:
                                st.session_state.last_sources = []
                                st.session_state.last_retrieved_docs = []

                        except Exception as e:
                            # Handle retrieval errors gracefully
                            logger.error(f"Error during document retrieval: {e}")
                            context = ""
                            st.session_state.last_sources = []
                            st.session_state.last_retrieved_docs = []

                            # Show user-friendly error message
                            st.warning(
                                "⚠️ Unable to retrieve documents from the knowledge base. "
                                "Answering based on general knowledge instead."
                            )
                    else:
                        # RAG is disabled or retrieval unavailable - use empty context
                        context = ""
                        st.session_state.last_sources = []
                        st.session_state.last_retrieved_docs = []

                    # Generate answer with available context (RAG-only mode)
                    try:
                        answer = process_user_prompt(model, prompt, context, message_history)

                        # Filter sources to only those cited in the answer
                        if st.session_state.last_sources and st.session_state.last_retrieved_docs:

                            # Find all citation numbers in the answer (e.g., [1], [2, 3], [1, 4, 5])
                            citation_pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
                            matches = re.findall(citation_pattern, answer)

                            # Extract all unique cited numbers
                            cited_numbers = set()
                            for match in matches:
                                numbers = [int(n.strip()) for n in match.split(',')]
                                cited_numbers.update(numbers)

                            if cited_numbers:
                                # Get sources that were actually cited (1-indexed)
                                all_sources = st.session_state.last_sources
                                cited_sources = []
                                missing_citations = []

                                for num in sorted(cited_numbers):
                                    if 1 <= num <= len(all_sources):
                                        cited_sources.append(all_sources[num - 1])
                                    else:
                                        missing_citations.append(num)

                                if missing_citations:
                                    logger.warning(
                                        f"LLM cited sources {missing_citations} but only {len(all_sources)} sources available"
                                    )

                                # Update last_sources to only include cited sources
                                # Keep all sources if filtering would result in empty list
                                if cited_sources:
                                    st.session_state.last_sources = cited_sources
                                else:
                                    logger.warning("No valid cited sources found, keeping all sources")

                        st.session_state.last_web_sources = []

                    except Exception as e:
                        logger.error(f"Error generating answer: {e}")
                        answer = "I apologize, but I encountered an error while generating a response. Please try again."
                        st.error("❌ Failed to generate response. Please try asking your question again.")
                        # Clear sources on error
                        st.session_state.last_sources = []

            except TimeoutError:
                logger.warning("Request timed out")
                answer = "I apologize, but the request timed out. Please try again."
                st.session_state.last_sources = []
                st.session_state.last_web_sources = []
            except Exception as e:
                logger.error(f"Unexpected error in processing: {e}", exc_info=True)
                answer = "I apologize, but an unexpected error occurred. Please try again or contact support if the issue persists."
                st.session_state.last_sources = []
                st.session_state.last_web_sources = []

            sys_message = {
                "role": "ai",
                "answer": answer,
                "sources": st.session_state.last_sources,
                "web_sources": st.session_state.get("last_web_sources", []),
            }
        display_message(sys_message)
        st.session_state.messages.append(sys_message)


if __name__ == "__main__":
    main()
