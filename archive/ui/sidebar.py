"""Sidebar component for the Streamlit app."""
import streamlit as st

from src.utils import get_config, get_initial_message


def render_sidebar(retriever=None, chroma_client=None):
    """
    Render the sidebar with app information and RAG settings.

    Args:
        retriever: ChromaRetriever instance (optional)
        chroma_client: ChromaDB client instance (optional)
    """
    cfg = get_config()

    with st.sidebar:
        st.write(
            """
            Pour Decisions uses Retrieval-Augmented Generation (RAG) and LLMs 
            to answer your wine-related questions using both curated knowledge and external wine data. 🍇
            """
        )
        st.markdown("---")

        # Agent Mode Selection
        st.subheader("🤖 Agent Mode")

        # Initialize session state for agent mode if not exists
        if "agent_mode" not in st.session_state:
            st.session_state.agent_mode = "Intelligent Agent"

        agent_mode = st.selectbox(
            "Select Agent Type",
            options=["Intelligent Agent", "Keyword Agent", "No Agent (RAG Only)"],
            index=["Intelligent Agent", "Keyword Agent", "No Agent (RAG Only)"].index(st.session_state.agent_mode),
            help="""
            - **Intelligent Agent**: Uses LLM to intelligently select and chain tools. Best for complex queries.
            - **Keyword Agent**: Uses pattern matching for routing. Faster, uses fewer LLM calls, ideal for testing.
            - **No Agent (RAG Only)**: Traditional RAG without agents. Uses only wine knowledge retrieval.
            """
        )
        st.session_state.agent_mode = agent_mode

        # RAG Settings Section (only for No Agent mode)
        if agent_mode == "No Agent (RAG Only)":
            st.subheader("⚙️ RAG Settings")

            # Show system status
            if retriever is not None and chroma_client is not None:
                st.success("✅ RAG System: Connected")
            else:
                st.error("❌ RAG System: Unavailable")
                st.caption("Using LLM general knowledge only")

            # Initialize session state for RAG toggle if not exists
            if "enable_rag" not in st.session_state:
                st.session_state.enable_rag = True

            # Toggle to enable/disable RAG retrieval (disabled if retrieval unavailable)
            enable_rag = st.toggle(
                "Enable RAG Retrieval",
                value=st.session_state.enable_rag if retriever is not None else False,
                help="When enabled, answers are based on your wine book collection. "
                     "When disabled, uses only the LLM's general knowledge.",
                disabled=(retriever is None)
            )
            st.session_state.enable_rag = enable_rag

            if enable_rag and retriever is not None:
                # Show number of chunks to retrieve
                n_results = st.slider(
                    "Number of sources to retrieve",
                    min_value=1,
                    max_value=10,
                    value=cfg.chroma.retrieval.n_results,
                    help="How many document chunks to retrieve from the knowledge base"
                )
                st.session_state.n_results = n_results


        if st.button("🔄 Reset Chat"):
            st.session_state.messages = get_initial_message()
            for key in ("last_sources", "last_retrieved_docs", "last_web_sources"):
                st.session_state.pop(key, None)
            st.rerun()
