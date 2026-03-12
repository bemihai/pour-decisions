"""Wine Cellar page UI."""
import os
import streamlit as st
from dotenv import load_dotenv

from src.ui.helper import (show_cellar_metrics, make_compact_page_title, show_cellar_inventory,
                           show_cellar_statistics, TABS_DISPLAY)
from src.etl.cellartracker_importer import CellarTrackerImporter
from src.utils import get_default_db_path
from src.utils.logger import logger


def sync_cellartracker_data():
    """Sync cellar-data from CellarTracker."""
    load_dotenv()

    username = os.getenv('CELLAR_TRACKER_USERNAME')
    password = os.getenv('CELLAR_TRACKER_PASSWORD')

    if not username or not password:
        st.session_state.sync_error = "❌ CellarTracker credentials not found! Please set CELLAR_TRACKER_USERNAME and CELLAR_TRACKER_PASSWORD in your .env file."
        st.session_state.sync_success = False
        return False

    try:
        with st.spinner("🔄 Syncing cellar-data from CellarTracker..."):
            db_path = get_default_db_path()
            importer = CellarTrackerImporter(username, password, db_path)

            stats = importer.import_all()

            # Store stats in session state to persist them
            st.session_state.last_sync_stats = stats
            st.session_state.sync_success = True
            st.session_state.sync_error = None

            return True

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        st.session_state.sync_error = f"❌ Sync failed: {str(e)}"
        st.session_state.sync_success = False
        return False


def main():
    """Wine Cellar page - main entry point."""
    st.set_page_config(page_title="Wine Cellar", page_icon="🍷", layout="wide")
    st.markdown(TABS_DISPLAY, unsafe_allow_html=True)

    # Initialize session state for sync stats
    if 'last_sync_stats' not in st.session_state:
        st.session_state.last_sync_stats = None
    if 'sync_success' not in st.session_state:
        st.session_state.sync_success = False
    if 'sync_error' not in st.session_state:
        st.session_state.sync_error = None
    if 'desc_generation_stats' not in st.session_state:
        st.session_state.desc_generation_stats = None

    # Sidebar with sync button
    with st.sidebar:
        st.markdown("### 🔄 Data Sync")
        st.markdown("")

        if st.button("Sync CellarTracker", type="primary", width="stretch"):
            sync_cellartracker_data()
            st.rerun()

        st.markdown("")
        st.caption("Manually sync your CellarTracker cellar-data to update your collection.")

        # Display sync results if available
        if st.session_state.sync_error:
            st.error(st.session_state.sync_error)

        if st.session_state.sync_success and st.session_state.last_sync_stats:
            stats = st.session_state.last_sync_stats

            st.success("✅ Sync completed!")

            st.markdown("---")
            st.markdown("#### Last Sync Summary")

            # Display summary
            st.metric("Wines", stats['wines_processed'],
                     delta=f"+{stats['wines_imported']} new")

            st.metric("Bottles", stats['bottles_processed'],
                     delta=f"+{stats['bottles_imported']} new")

            st.metric("Producers", stats['producers_created'],
                     delta="created")

            st.metric("Regions", stats['regions_created'],
                     delta="created")

            if stats['errors']:
                st.warning(f"⚠️ {len(stats['errors'])} errors")
                with st.expander("View Errors"):
                    for error in stats['errors']:
                        st.code(error, language=None)

        # Description Generation Section
        st.markdown("---")
        st.markdown("### ✨ AI Descriptions")
        st.markdown("")

        # RAG Settings
        with st.expander("⚙️ Settings", expanded=False):
            # Check ChromaDB availability
            chromadb_available = False
            try:
                from pathlib import Path
                chroma_db_path = Path(__file__).parent.parent.parent.parent / "chroma-data"
                chromadb_available = chroma_db_path.exists() and (chroma_db_path / "chroma.sqlite3").exists()
            except Exception:
                chromadb_available = False

            # RAG toggle
            if chromadb_available:
                use_rag = st.checkbox(
                    "Use Wine Books Context",
                    value=st.session_state.get('use_rag_context', False),
                    help="Generate descriptions using wine book knowledge for better accuracy (may be slower)"
                )
                st.session_state['use_rag_context'] = use_rag

                if use_rag:
                    st.success("📚 Wine books enabled")
                else:
                    st.info("🤖 Using AI general knowledge")
            else:
                st.warning("⚠️ ChromaDB not available")
                st.caption("Wine books context disabled. Using AI general knowledge only.")
                st.session_state['use_rag_context'] = False

            st.markdown("---")
            use_web_search = st.checkbox(
                "Use Web Search Context",
                value=st.session_state.get('use_web_search', False),
                help="Inject web search snippets about aging potential into the LLM prompt. Uses Tavily free tier (1000 searches/month)."
            )
            st.session_state['use_web_search'] = use_web_search
            if use_web_search:
                st.caption("Web search context enabled. Uses Tavily free tier.")

        st.markdown("")

        # Show description statistics
        from src.database.repository import WineRepository, ProducerRepository
        wine_repo = WineRepository()
        producer_repo = ProducerRepository()

        wine_stats = wine_repo.count_with_description()
        producer_stats = producer_repo.count_with_description()

        st.caption(f"Wines: {wine_stats['with_description']}/{wine_stats['total']}")
        st.caption(f"Producers: {producer_stats['with_description']}/{producer_stats['total']}")

        # Batch generation button
        wines_to_generate = wine_stats['without_description']
        producers_to_generate = producer_stats['without_description']
        total_to_generate = wines_to_generate + producers_to_generate

        if total_to_generate > 0:
            button_label = f"Generate {min(total_to_generate, 10)} Descriptions"
            if st.button(button_label, type="secondary", width="stretch"):
                with st.spinner(f"Generating descriptions (this may take a minute)..."):
                    try:
                        from src.agents.description_service import get_description_service

                        use_rag = st.session_state.get('use_rag_context', False)
                        use_web = st.session_state.get('use_web_search', False)
                        service = get_description_service(use_rag_context=use_rag, use_web_search=use_web)

                        # Get items without descriptions (limit to 10 for quick batch)
                        wines_batch = wine_repo.get_without_description(limit=5)
                        producers_batch = producer_repo.get_without_description(limit=5)

                        # Generate descriptions
                        result = service.generate_batch(
                            producers=producers_batch,
                            wines=wines_batch
                        )

                        st.session_state.desc_generation_stats = result
                        st.rerun()

                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
        else:
            st.success("✅ All items have descriptions!")

        # Show generation results if available
        if st.session_state.desc_generation_stats:
            stats = st.session_state.desc_generation_stats
            st.markdown("---")
            st.markdown("#### Last Generation")
            st.metric("Wines", stats['wines_generated'])
            st.metric("Producers", stats['producers_generated'])


    # Header
    st.markdown(make_compact_page_title(
        "Cellar",
        "Your personal wine collection"
    ), unsafe_allow_html=True)
    st.markdown("")

    with st.container(border=True):
        show_cellar_metrics()

    tab_1, tab_2 = st.tabs(["Cellar Inventory", "Statistics & Charts"])

    with tab_1:
        with st.container():
            show_cellar_inventory()

    with tab_2:
        with st.container():
            show_cellar_statistics()


if __name__ == "__main__":
    main()

