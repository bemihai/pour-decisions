"""Helper functions to display cellar statistics and inventory in Streamlit UI."""
import math

import streamlit as st
from src.database import get_db_connection
from src.database.repository import StatsRepository, BottleRepository
from src.etl.utils import denormalize_rating, get_rating_description
from src.ui.helper.display import render_drinking_index_bar


def _generate_description(entity_type: str, entity_id: int, repository_class, service_method_name: str, success_message: str) -> None:
    """Generate, persist, and display description for wine or producer entity.
    
    This function handles the complete workflow: retrieves the entity from the repository,
    generates a description using the LLM service, persists it, displays success/error
    feedback to the user, and triggers a UI rerun to show the new description.
    
    Args:
        entity_type: Type of entity ('Wine' or 'Producer')
        entity_id: ID of the entity
        repository_class: Repository class to instantiate (WineRepository or ProducerRepository)
        service_method_name: Name of the service method to call ('get_wine_description' or 'get_producer_description')
        success_message: Message to display on success
    """
    from src.agents.description_service import get_description_service
    
    use_rag = st.session_state.get('use_rag_context', False)
    service = get_description_service(use_rag_context=use_rag)
    repo = repository_class()
    
    entity = repo.get_by_id(entity_id)
    if entity:
        service_method = getattr(service, service_method_name)
        description = service_method(entity)
        if description:
            st.success(success_message)
            st.rerun()
        else:
            st.error(f"Failed to generate {entity_type.lower()} description")
    else:
        st.error(f"{entity_type} not found")


def _regenerate_description_with_confirmation(
    entity_type: str, 
    entity_id: int, 
    repository_class, 
    service_method_name: str, 
    confirmation_key: str,
    success_message: str,
    spinner_message: str
) -> None:
    """Regenerate description with two-step confirmation flow.
    
    This function implements a confirmation mechanism to prevent accidental regeneration:
    - First call: Sets a confirmation flag in session state and shows a warning
    - Second call: Clears existing description, regenerates it, displays feedback, and triggers UI rerun
    
    The spinner is shown only during actual regeneration work (second call).
    
    Args:
        entity_type: Type of entity ('Wine' or 'Producer')
        entity_id: ID of the entity
        repository_class: Repository class to instantiate (WineRepository or ProducerRepository)
        service_method_name: Name of the service method to call ('get_wine_description' or 'get_producer_description')
        confirmation_key: Session state key for confirmation tracking
        success_message: Message to display on success
        spinner_message: Message to display in spinner during regeneration
    """
    from src.agents.description_service import get_description_service
    
    if st.session_state.get(confirmation_key):
        # Confirmed - regenerate with spinner
        with st.spinner(spinner_message):
            use_rag = st.session_state.get('use_rag_context', False)
            service = get_description_service(use_rag_context=use_rag)
            repo = repository_class()
            
            # Clear existing description
            repo.update_description(entity_id, None)
            
            # Get full entity object and regenerate
            entity = repo.get_by_id(entity_id)
            if entity:
                service_method = getattr(service, service_method_name)
                description = service_method(entity)
                if description:
                    st.success(success_message)
                    st.session_state[confirmation_key] = False
                    st.rerun()
                else:
                    st.error(f"Failed to regenerate {entity_type.lower()} description")
            else:
                st.error(f"{entity_type} not found")
    else:
        # First click - ask for confirmation
        st.session_state[confirmation_key] = True
        st.warning("Click again to confirm regeneration")


def show_cellar_metrics():
    """Display key cellar metrics in a row of streamlit metrics."""
    stats_repo = StatsRepository()

    overview = stats_repo.get_cellar_overview()
    drinking_stats = stats_repo.get_drinking_window_stats()
    value_stats = stats_repo.get_cellar_value()

    # Add a title for the metrics section
    st.markdown("### <i class='fa-solid fa-chart-bar fa-icon'></i>Cellar Overview", unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Bottles",
            value=f"{overview['total_bottles']:,}",
            delta=f"{overview['unique_wines']} unique wines"
        )

    with col2:
        top_type = overview["by_type"][0]
        percentage = (top_type["bottles"] / overview["total_bottles"] * 100) if overview["total_bottles"] > 0 else 0
        st.metric(
            label=f"{top_type['wine_type']}",
            value=f"{percentage:.1f}%",
            delta=f"{top_type['bottles']} bottles"
        )

    with col3:
        st.metric(label="Ready to Drink", value=f"{drinking_stats['ready_to_drink']:,}")

    with col4:
        st.metric(label="To Hold", value=f"{drinking_stats['to_hold']:,}")

    with col5:
        primary = value_stats["by_currency"][0]
        st.metric(
            label="Cellar Value",
            value=f"{primary['currency']} {primary['total_value']:,.0f}"
        )


def show_top_rated_consumed_wines():
    """Display the top 5 rated wines consumed by the user."""
    state_repo = StatsRepository()

    # Get top 5 consumed bottles with ratings
    top_wines = state_repo.get_consumed_with_ratings(limit=5)

    if not top_wines:
        st.warning("No consumed wines with ratings found yet.")
        return

    st.markdown("### <i class='fa-solid fa-star fa-icon'></i>Top 5 Rated Wines", unsafe_allow_html=True)
    st.markdown("")

    for idx, wine_data in enumerate(top_wines, 1):
        rating = wine_data.get("personal_rating", 0)
        wine_name = wine_data.get("wine_name", "Unknown")
        producer_name = wine_data.get("producer_name", "Unknown Producer")
        vintage = wine_data.get("vintage")
        wine_type = wine_data.get("wine_type", "Unknown")
        consumed_date = wine_data.get("consumed_date", "Unknown date")

        # Create star rating display using Font Awesome
        denorm_rating = denormalize_rating(rating)
        stars_html = ""
        if denorm_rating:
            full_stars = int(denorm_rating)
            stars_html = f"<i class='fa-solid fa-star' style='color: #FFD700;'></i> " * full_stars

        with st.expander(f"{producer_name} {wine_name} ({vintage or 'NV'}) - {rating}/100"):
            if stars_html:
                st.markdown(f"**Rating:** {rating}/100 {stars_html}", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Producer:** {producer_name}")
                st.write(f"**Wine:** {wine_name}")
                st.write(f"**Vintage:** {vintage or 'Non-Vintage'}")
                st.write(f"**Type:** {wine_type}")

            with col2:
                st.write(f"**Consumed:** {consumed_date}")
                tasting_notes = wine_data.get("tasting_notes", "")
                if tasting_notes:
                    st.write(f"**Notes:** {tasting_notes}")


def show_latest_consumed_wines(limit: int = 5):
    """Display the latest consumed wines by the user."""

    # Get latest consumed bottles
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                b.*,
                w.wine_name, w.wine_type, w.vintage,
                p.name as producer_name,
                r.country, 
                COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                t.personal_rating,
                t.tasting_notes
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN producers p ON w.producer_id = p.id
            LEFT JOIN regions r ON w.region_id = r.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed'
            ORDER BY b.consumed_date DESC
            LIMIT ?
        """, (limit,))
        latest_wines = [dict(row) for row in cursor.fetchall()]

    if not latest_wines:
        st.warning("No consumed wines found yet.")
        return

    st.markdown("### <i class='fa-solid fa-clock fa-icon'></i>Latest 5 Consumed", unsafe_allow_html=True)
    st.markdown("")

    for idx, wine_data in enumerate(latest_wines, 1):
        rating = wine_data.get("personal_rating")
        wine_name = wine_data.get("wine_name", "Unknown")
        producer_name = wine_data.get("producer_name", "Unknown Producer")
        vintage = wine_data.get("vintage")
        wine_type = wine_data.get("wine_type", "Unknown")
        consumed_date = wine_data.get("consumed_date", "Unknown date")

        # Create title with rating if available
        if rating:
            title = f"{producer_name} {wine_name} ({vintage or 'NV'}) - {rating}/100"
        else:
            title = f"{producer_name} {wine_name} ({vintage or 'NV'})"

        with st.expander(f"{title}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Producer:** {producer_name}")
                st.write(f"**Wine:** {wine_name}")
                st.write(f"**Vintage:** {vintage or 'Non-Vintage'}")
                st.write(f"**Type:** {wine_type}")

            with col2:
                st.write(f"**Consumed:** {consumed_date}")
                if rating:
                    denorm_rating = denormalize_rating(rating)
                    stars_html = ""
                    if denorm_rating:
                        full_stars = int(denorm_rating)
                        stars_html = f"<i class='fa-solid fa-star' style='color: #FFD700;'></i> " * full_stars
                    st.markdown(f"**Rating:** {rating}/100 {stars_html}", unsafe_allow_html=True)

                tasting_notes = wine_data.get("tasting_notes", "")
                if tasting_notes:
                    st.write(f"**Notes:** {tasting_notes}")


def show_top_rated_consumed_wines_old():
    """Display the top 5 rated wines consumed by the user."""
    state_repo = StatsRepository()

    # Get top 5 consumed bottles with ratings
    top_wines = state_repo.get_consumed_with_ratings(limit=5)

    if not top_wines:
        st.warning("No consumed wines with ratings found yet.")
        return

    st.markdown("### <i class='fa-solid fa-star fa-icon'></i>Your Top 5 Rated Consumed Wines", unsafe_allow_html=True)
    st.markdown("")

    for idx, wine_data in enumerate(top_wines, 1):
        rating = wine_data.get("personal_rating", 0)
        wine_name = wine_data.get("wine_name", "Unknown")
        producer_name = wine_data.get("producer_name", "Unknown Producer")
        vintage = wine_data.get("vintage")
        wine_type = wine_data.get("wine_type", "Unknown")
        country = wine_data.get("country", "Unknown")
        region_name = wine_data.get("region_name", "")
        consumed_date = wine_data.get("consumed_date", "Unknown date")
        bottle_note = wine_data.get("bottle_note", "")
        tasting_notes = wine_data.get("tasting_notes", "")

        # Create star rating display using Font Awesome
        denorm_rating = denormalize_rating(rating)
        stars_html = ""
        if denorm_rating:
            full_stars = int(denorm_rating)
            stars_html = f"<i class='fa-solid fa-star' style='color: #FFD700;'></i> " * full_stars

        with st.expander(f"#{idx} - {producer_name} {wine_name} ({vintage or 'NV'}) - {rating}/100"):
            if stars_html:
                st.markdown(f"**Rating:** {rating}/100 {stars_html}", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Producer:** {producer_name}")
                st.write(f"**Wine:** {wine_name}")
                st.write(f"**Vintage:** {vintage or 'Non-Vintage'}")
                st.write(f"**Type:** {wine_type}")

            with col2:
                st.write(f"**Country:** {country}")
                if region_name:
                    st.write(f"**Region:** {region_name}")
                st.write(f"**Consumed:** {consumed_date}")

            # Show tasting notes from wine or bottle
            if tasting_notes:
                st.markdown(f"**Tasting Notes:** {tasting_notes}")
            elif bottle_note:
                st.markdown(f"**Bottle Notes:** {bottle_note}")


def show_cellar_inventory():
    """Display all wines currently in the cellar."""
    bottle_repo = BottleRepository()

    # Get all inventory for filter options
    raw_inventory = bottle_repo.get_inventory()

    # Group bottles by wine_id and aggregate quantities
    wine_groups = {}
    for bottle in raw_inventory:
        wine_id = bottle.get('wine_id')
        if wine_id not in wine_groups:
            # First occurrence - store all wine cellar-data
            wine_groups[wine_id] = bottle.copy()
        else:
            # Add quantity to existing group
            wine_groups[wine_id]['quantity'] = wine_groups[wine_id].get('quantity', 0) + bottle.get('quantity', 0)

    # Convert back to list
    all_inventory = list(wine_groups.values())

    # Extract unique values for filters
    wine_types = sorted(set(w.get('wine_type') for w in all_inventory if w.get('wine_type')))
    countries = sorted(set(w.get('country') for w in all_inventory if w.get('country')))
    locations = sorted(set(w.get('location') for w in all_inventory if w.get('location')))
    producers = sorted(set(w.get('producer_name') for w in all_inventory if w.get('producer_name')))

    # Get vintage range
    vintages = [w.get('vintage') for w in all_inventory if w.get('vintage')]
    min_vintage = min(vintages) if vintages else 2000
    max_vintage = max(vintages) if vintages else 2024

    # Create filter UI with bordered container
    with st.container(border=True):
        st.markdown("### <i class='fa-solid fa-filter fa-icon'></i>Filter Your Collection", unsafe_allow_html=True)
        st.markdown("")  # Add spacing

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            wine_types_with_all = ["All Types"] + wine_types
            selected_type = st.selectbox("Wine Type", wine_types_with_all)

        with filter_col2:
            countries_with_all = ["All Countries"] + countries
            selected_country = st.selectbox("Country", countries_with_all)

        with filter_col3:
            locations_with_all = ["All Locations"] + locations
            selected_location = st.selectbox("Location", locations_with_all)

        with filter_col4:
            producers_with_all = ["All Producers"] + producers
            selected_producer = st.selectbox("Producer", producers_with_all)

        # Additional filters row
        filter_col5, filter_col6, filter_col7, filter_col8 = st.columns(4)

        with filter_col5:
            vintage_range = st.slider(
                "Vintage Range",
                min_value=int(min_vintage),
                max_value=int(max_vintage),
                value=(int(min_vintage), int(max_vintage))
            )

        with filter_col6:
            rating_filter = st.selectbox("Rating", ["All Ratings", "Rated Only", "Unrated", "90+", "80+", "70+"])

        with filter_col7:
            search_term = st.text_input("Search", placeholder="Wine name, varietal...")

        with filter_col8:
            sort_by = st.selectbox("Sort By", ["Producer", "Wine Name", "Vintage (New→Old)", "Vintage (Old→New)", "Rating (High→Low)", "Rating (Low→High)", "Drink (Sooner->Later)", "Drink (Later->Sooner)"])

    # Apply filters
    filtered_inventory = all_inventory

    # Filter by wine type
    if selected_type != "All Types":
        filtered_inventory = [w for w in filtered_inventory if w.get('wine_type') == selected_type]

    # Filter by country
    if selected_country != "All Countries":
        filtered_inventory = [w for w in filtered_inventory if w.get('country') == selected_country]

    # Filter by location
    if selected_location != "All Locations":
        filtered_inventory = [w for w in filtered_inventory if w.get('location') == selected_location]

    # Filter by producer
    if selected_producer != "All Producers":
        filtered_inventory = [w for w in filtered_inventory if w.get('producer_name') == selected_producer]

    # Filter by vintage range
    filtered_inventory = [
        w for w in filtered_inventory
        if w.get('vintage') is None or (vintage_range[0] <= w.get('vintage') <= vintage_range[1])
    ]

    # Filter by rating
    if rating_filter == "Rated Only":
        filtered_inventory = [w for w in filtered_inventory if w.get('personal_rating') is not None]
    elif rating_filter == "Unrated":
        filtered_inventory = [w for w in filtered_inventory if w.get('personal_rating') is None]
    elif rating_filter == "90+":
        filtered_inventory = [w for w in filtered_inventory if w.get('personal_rating', 0) >= 90]
    elif rating_filter == "80+":
        filtered_inventory = [w for w in filtered_inventory if w.get('personal_rating', 0) >= 80]
    elif rating_filter == "70+":
        filtered_inventory = [w for w in filtered_inventory if w.get('personal_rating', 0) >= 70]

    # Filter by search term
    if search_term:
        search_lower = search_term.lower()
        filtered_inventory = [
            w for w in filtered_inventory
            if search_lower in w.get('wine_name', '').lower()
            or search_lower in w.get('producer_name', '').lower()
        ]

    # Sort
    if sort_by == "Producer":
        filtered_inventory.sort(key=lambda w: (w.get('producer_name', ''), w.get('vintage') or 0))
    elif sort_by == "Wine Name":
        filtered_inventory.sort(key=lambda w: w.get('wine_name', ''))
    elif sort_by == "Vintage (New→Old)":
        filtered_inventory.sort(key=lambda w: w.get('vintage') or 0, reverse=True)
    elif sort_by == "Vintage (Old→New)":
        filtered_inventory.sort(key=lambda w: w.get('vintage') or 9999)
    elif sort_by == "Rating (High→Low)":
        filtered_inventory.sort(key=lambda w: w.get('personal_rating') or 0, reverse=True)
    elif sort_by == "Rating (Low→High)":
        filtered_inventory.sort(key=lambda w: w.get('personal_rating') or 9999)
    elif sort_by == "Drink (Sooner->Later)":
        filtered_inventory.sort(key=lambda w: w.get('drink_index') or 0, reverse=True)
    elif sort_by == "Drink (Later->Sooner)":
        filtered_inventory.sort(key=lambda w: w.get('drink_index') or -9999)

    if not filtered_inventory:
        st.warning("No wines found matching the selected filters.")
        return

    # Results header
    total_bottles = sum(w.get('quantity', 0) for w in filtered_inventory)
    st.markdown("")  # Add spacing
    st.markdown(f"### Your Collection ({len(filtered_inventory)} wines, {total_bottles} bottles)", unsafe_allow_html=True)
    st.markdown("")  # Add spacing before wine cards

    # Display wines in expandable sections
    for wine_data in filtered_inventory:
        wine_name = wine_data.get('wine_name', 'Unknown')
        producer_name = wine_data.get('producer_name', 'Unknown Producer')
        vintage = wine_data.get('vintage')
        wine_type = wine_data.get('wine_type', 'Unknown')
        country = wine_data.get('country', 'Unknown')
        region_name = wine_data.get('region_name', '')
        quantity = wine_data.get('quantity', 0)
        location = wine_data.get('location', 'Unknown')
        bin_location = wine_data.get('bin', '')
        purchase_date = wine_data.get('purchase_date', '')
        purchase_price = wine_data.get('purchase_price')
        currency = wine_data.get('currency', 'RON')
        rating = wine_data.get('personal_rating')
        bottle_note = wine_data.get('bottle_note', '')

        # Create title with rating if available
        title_parts = [f"{producer_name}, {wine_name} ({vintage or 'NV'})"]
        if rating:
            title_parts.append(f"- {quantity} bottle{'s' if quantity > 1 else ''} - {rating}/100")
        else:
            title_parts.append(f"- {quantity} bottle{'s' if quantity > 1 else ''}")

        with st.expander(" ".join(title_parts)):
            # Display wine and producer descriptions in two columns
            wine_description = wine_data.get('description')
            producer_description = wine_data.get('producer_description')

            # Check if we have any descriptions to show
            has_wine_desc = wine_description is not None
            has_producer_desc = producer_description is not None

            if has_wine_desc or has_producer_desc:
                # Two-column layout for descriptions with smaller text
                desc_col1, desc_col2 = st.columns(2)

                with desc_col1:
                    # Header with regenerate button
                    header_col1, header_col2 = st.columns([5, 1])
                    with header_col1:
                        st.markdown("<small><b>About this Wine</b></small>", unsafe_allow_html=True)
                    with header_col2:
                        if has_wine_desc:
                            wine_id = wine_data.get('wine_id')
                            if wine_id and st.button("🔄", key=f"regen_wine_{wine_id}", help="Regenerate wine description"):
                                try:
                                    from src.database.repository import WineRepository
                                    _regenerate_description_with_confirmation(
                                        "Wine", 
                                        wine_id, 
                                        WineRepository, 
                                        "get_wine_description",
                                        f'confirm_regen_wine_{wine_id}',
                                        "Wine description regenerated!",
                                        "Regenerating wine description..."
                                    )
                                except Exception as e:
                                    st.session_state[f'confirm_regen_wine_{wine_id}'] = False
                                    st.error(f"Error: {str(e)}")
                                if st.session_state.get(f'confirm_regen_wine_{wine_id}'):
                                    # Confirmed - regenerate
                                    with st.spinner("Regenerating wine description..."):
                                        try:
                                            from src.agents.description_service import get_description_service
                                            from src.database.repository import WineRepository

                                            use_rag = st.session_state.get('use_rag_context', False)
                                            service = get_description_service(use_rag_context=use_rag)
                                            wine_repo = WineRepository()

                                            # Clear existing description
                                            wine_repo.update_description(wine_id, None)

                                            # Get full wine object and regenerate
                                            wine = wine_repo.get_by_id(wine_id)
                                            if wine:
                                                description = service.get_wine_description(wine)
                                                if description:
                                                    st.success("Wine description regenerated!")
                                                    st.session_state[f'confirm_regen_wine_{wine_id}'] = False
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to regenerate wine description")
                                            else:
                                                st.error("Wine not found")
                                        except Exception as e:
                                            st.session_state[f'confirm_regen_wine_{wine_id}'] = False
                                            st.error(f"Error: {str(e)}")
                                else:
                                    # First click - ask for confirmation
                                    st.session_state[f'confirm_regen_wine_{wine_id}'] = True
                                    st.warning("Click again to confirm regeneration")

                    if has_wine_desc:
                        st.markdown(f"<small><i>{wine_description}</i></small>", unsafe_allow_html=True)
                        # Reset confirmation state if description is shown
                        if st.session_state.get(f'confirm_regen_wine_{wine_id}'):
                            if st.button("Cancel", key=f"cancel_regen_wine_{wine_id}"):
                                st.session_state[f'confirm_regen_wine_{wine_id}'] = False
                                st.rerun()
                    else:
                        wine_id = wine_data.get('wine_id')
                        if wine_id and st.button(f"✨ Generate Wine Description", key=f"gen_wine_desc_{wine_id}", type="secondary"):
                            with st.spinner("Generating wine description..."):
                                try:
                                    from src.database.repository import WineRepository
                                    _generate_description("Wine", wine_id, WineRepository, "get_wine_description", "Wine description generated!")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                with desc_col2:
                    # Header with regenerate button
                    header_col1, header_col2 = st.columns([5, 1])
                    with header_col1:
                        st.markdown("<small><b>About the Producer</b></small>", unsafe_allow_html=True)
                    with header_col2:
                        if has_producer_desc:
                            producer_id = wine_data.get('producer_id')
                            wine_id = wine_data.get('wine_id')
                            if producer_id and st.button("🔄", key=f"regen_prod_{producer_id}_{wine_id}", help="Regenerate producer description"):
                                try:
                                    from src.database.repository import ProducerRepository
                                    _regenerate_description_with_confirmation(
                                        "Producer",
                                        producer_id,
                                        ProducerRepository,
                                        "get_producer_description",
                                        f'confirm_regen_prod_{producer_id}_{wine_id}',
                                        "Producer description regenerated!",
                                        "Regenerating producer description..."
                                    )
                                except Exception as e:
                                    st.session_state[f'confirm_regen_prod_{producer_id}_{wine_id}'] = False
                                    st.error(f"Error: {str(e)}")
                                if st.session_state.get(f'confirm_regen_prod_{producer_id}_{wine_id}'):
                                    # Confirmed - regenerate
                                    with st.spinner("Regenerating producer description..."):
                                        try:
                                            from src.agents.description_service import get_description_service
                                            from src.database.repository import ProducerRepository

                                            use_rag = st.session_state.get('use_rag_context', False)
                                            service = get_description_service(use_rag_context=use_rag)
                                            producer_repo = ProducerRepository()

                                            # Clear existing description
                                            producer_repo.update_description(producer_id, None)

                                            # Get full producer object and regenerate
                                            producer = producer_repo.get_by_id(producer_id)
                                            if producer:
                                                description = service.get_producer_description(producer)
                                                if description:
                                                    st.success("Producer description regenerated!")
                                                    st.session_state[f'confirm_regen_prod_{producer_id}_{wine_id}'] = False
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to regenerate producer description")
                                            else:
                                                st.error("Producer not found")
                                        except Exception as e:
                                            st.session_state[f'confirm_regen_prod_{producer_id}_{wine_id}'] = False
                                            st.error(f"Error: {str(e)}")
                                else:
                                    # First click - ask for confirmation
                                    st.session_state[f'confirm_regen_prod_{producer_id}_{wine_id}'] = True
                                    st.warning("Click again to confirm regeneration")

                    if has_producer_desc:
                        st.markdown(f"<small><i>{producer_description}</i></small>", unsafe_allow_html=True)
                        # Reset confirmation state if description is shown
                        if st.session_state.get(f'confirm_regen_prod_{producer_id}_{wine_id}'):
                            if st.button("Cancel", key=f"cancel_regen_prod_{producer_id}_{wine_id}"):
                                st.session_state[f'confirm_regen_prod_{producer_id}_{wine_id}'] = False
                                st.rerun()
                    else:
                        producer_id = wine_data.get('producer_id')
                        wine_id = wine_data.get('wine_id')
                        if producer_id and st.button(f"✨ Generate Producer Description", key=f"gen_prod_desc_{producer_id}_{wine_id}", type="secondary"):
                            with st.spinner("Generating producer description..."):
                                try:
                                    from src.database.repository import ProducerRepository
                                    _generate_description("Producer", producer_id, ProducerRepository, "get_producer_description", "Producer description generated!")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
            else:
                # No descriptions - show generate buttons side by side
                gen_col1, gen_col2 = st.columns(2)

                with gen_col1:
                    wine_id = wine_data.get('wine_id')
                    if wine_id and st.button(f"✨ Generate Wine Description", key=f"gen_wine_desc_{wine_id}", type="secondary", width="stretch"):
                        with st.spinner("Generating wine description..."):
                            try:
                                from src.database.repository import WineRepository
                                _generate_description("Wine", wine_id, WineRepository, "get_wine_description", "Wine description generated!")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                with gen_col2:
                    producer_id = wine_data.get('producer_id')
                    wine_id = wine_data.get('wine_id')
                    if producer_id and st.button(f"✨ Generate Producer Description", key=f"gen_prod_desc_{producer_id}_{wine_id}", type="secondary", width="stretch"):
                        with st.spinner("Generating producer description..."):
                            try:
                                from src.database.repository import ProducerRepository
                                _generate_description("Producer", producer_id, ProducerRepository, "get_producer_description", "Producer description generated!")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")


            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Wine Details**")
                st.write(f"Producer: {producer_name}")
                st.write(f"Wine: {wine_name}")
                st.write(f"Vintage: {vintage or 'Non-Vintage'}")
                st.write(f"Type: {wine_type}")
                st.write(f"Country: {country}")
                if region_name:
                    st.write(f"Region: {region_name}")

            with col2:
                st.write("**Cellar Info**")
                st.write(f"Quantity: {quantity} bottle{'s' if quantity > 1 else ''}")
                # Merge bin into location
                location_text = location
                if bin_location:
                    location_text = f"{location}, Bin {bin_location}"
                st.write(f"Location: {location_text}")
                if purchase_date:
                    st.write(f"Purchased: {purchase_date}")
                if purchase_price:
                    st.write(f"Price: {purchase_price} {currency}")

                # Display drinking window if available
                drink_from = wine_data.get('drink_from_year')
                drink_to = wine_data.get('drink_to_year')
                if drink_from or drink_to:
                    from_str = str(drink_from) if drink_from else "Now"
                    to_str = str(drink_to) if drink_to else "∞"
                    st.write(f"Drinking Window: {from_str} - {to_str}")

                # Display drinking index if available with visual progress bar
                drink_index = wine_data.get('drink_index')
                if drink_index is not None:
                    # Get global min/max for all wines in inventory
                    all_indices = [w.get('drink_index') for w in filtered_inventory if w.get('drink_index') is not None]
                    if all_indices:
                        render_drinking_index_bar(drink_index, all_indices)


            with col3:
                st.write("**Rating & Notes**")
                if rating:
                    # Create Font Awesome stars
                    denorm_rating = denormalize_rating(rating)
                    stars_html = ""
                    if denorm_rating:
                        full_stars = math.ceil(denorm_rating)
                        stars_html = f"<i class='fa-solid fa-star' style='color: #FFD700;'></i> " * full_stars

                    st.markdown(f"Rating: {rating}/100 {stars_html}", unsafe_allow_html=True)
                    st.write(f"Category: {get_rating_description(rating)}")

                else:
                    st.write("Rating: Not rated")

                if bottle_note:
                    st.write(f"Notes: {bottle_note}")


def show_cellar_statistics():
    """Display comprehensive cellar statistics with charts."""
    stats_repo = StatsRepository()
    bottle_repo = BottleRepository()

    st.markdown("### <i class='fa-solid fa-chart-line fa-icon'></i>Cellar Statistics & Analytics", unsafe_allow_html=True)


    # Get cellar-data
    overview = stats_repo.get_cellar_overview()
    inventory = bottle_repo.get_inventory()
    drinking_window_wines = stats_repo.get_drinking_window_wines()

    # Key Insights at the top
    with st.container(border=True):
        st.markdown("#### Key Insights")
        insight_col1, insight_col2, insight_col3, insight_col4, insight_col5 = st.columns(5)

        with insight_col1:
            avg_vintage = sum(w.get('vintage', 0) * w.get('quantity', 0) for w in inventory if w.get('vintage')) / max(sum(w.get('quantity', 0) for w in inventory if w.get('vintage')), 1)
            st.metric("Average Vintage", f"{int(avg_vintage)}")

        with insight_col2:
            rated_wines = len([w for w in inventory if w.get('personal_rating')])
            total_wines = len(inventory)
            rating_pct = (rated_wines / total_wines * 100) if total_wines > 0 else 0
            st.metric("Rated Wines", f"{rating_pct:.0f}%", f"{rated_wines}/{total_wines}")

        with insight_col3:
            avg_rating = sum(w.get('personal_rating', 0) for w in inventory if w.get('personal_rating')) / max(len([w for w in inventory if w.get('personal_rating')]), 1)
            st.metric("Average Rating", f"{avg_rating:.1f}/100")

        with insight_col4:
            unique_producers = len(set(w.get('producer_name') for w in inventory if w.get('producer_name')))
            st.metric("Unique Producers", f"{unique_producers}")

        with insight_col5:
            # Calculate collection value per bottle
            total_value = sum(w.get('quantity', 0) * w.get('purchase_price', 0) for w in inventory if w.get('purchase_price'))
            total_bottles = sum(w.get('quantity', 0) for w in inventory if w.get('purchase_price'))
            avg_price = total_value / total_bottles if total_bottles > 0 else 0
            st.metric("Avg. Price/Bottle", f"{int(avg_price)} RON")

    # Row 1: Wine Type Distribution, Country Distribution, Vintage Distribution
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("#### Wine Type Distribution")
            if overview['by_type']:
                import plotly.graph_objects as go

                labels = [wt['wine_type'] for wt in overview['by_type']]
                values = [wt['bottles'] for wt in overview['by_type']]

                wine_colors = {
                    'Red': 'rgba(139, 26, 26, 0.85)',
                    'White': 'rgba(244, 229, 161, 0.85)',
                    'Rosé': 'rgba(255, 182, 193, 0.85)',
                    'Sparkling': 'rgba(255, 215, 0, 0.85)',
                    'Dessert': 'rgba(221, 161, 94, 0.85)',
                    'Fortified': 'rgba(160, 82, 45, 0.85)'
                }
                colors = [wine_colors.get(label, 'rgba(123, 31, 162, 0.85)') for label in labels]

                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors),
                    hole=0.4,
                    textinfo='label+percent',
                    textposition='auto'
                )])
                fig.update_layout(
                    showlegend=True,
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, width="stretch")

    with col2:
        with st.container(border=True):
            st.markdown("#### Country Distribution")
            if overview['by_country']:
                import plotly.graph_objects as go

                countries = [c['country'] if c['country'] else 'Unknown' for c in overview['by_country'][:8]]
                bottles = [c['bottles'] for c in overview['by_country'][:8]]

                fig = go.Figure(data=[go.Bar(
                    x=countries,
                    y=bottles,
                    marker_color='rgba(123, 31, 162, 0.85)',
                    text=bottles,
                    textposition='auto'
                )])
                fig.update_layout(
                    xaxis_title="Country",
                    yaxis_title="Bottles",
                    showlegend=False,
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, width="stretch")

    with col3:
        with st.container(border=True):
            st.markdown("#### Vintage Distribution")
            from collections import Counter
            vintage_counts = Counter()
            for wine in inventory:
                vintage = wine.get('vintage')
                if vintage:
                    vintage_counts[vintage] += wine.get('quantity', 0)

            if vintage_counts:
                import plotly.graph_objects as go

                vintages = sorted(vintage_counts.keys())
                counts = [vintage_counts[v] for v in vintages]

                n_bars = len(vintages)
                colors = [f'rgba({int(139 + i * (220-139)/max(n_bars-1, 1))}, {int(26 + i * (130-26)/max(n_bars-1, 1))}, {int(26 + i * (100-26)/max(n_bars-1, 1))}, 0.85)'
                         for i in range(n_bars)]

                fig = go.Figure(data=[go.Bar(
                    x=vintages,
                    y=counts,
                    marker_color=colors,
                    text=counts,
                    textposition='auto'
                )])
                fig.update_layout(
                    xaxis_title="Vintage Year",
                    yaxis_title="Bottles",
                    showlegend=False,
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, width="stretch")

    # Row 2: Rating Distribution, Drinking Window Status, Wine Age Analysis
    col4, col5, col6 = st.columns(3)

    with col4:
        with st.container(border=True):
            st.markdown("#### Rating Distribution")
            ratings = [w.get('personal_rating') for w in inventory if w.get('personal_rating') is not None]

            if ratings:
                import plotly.graph_objects as go

                rating_categories = {
                    'Exceptional (98-100)': len([r for r in ratings if r >= 98]),
                    'Outstanding (94-97)': len([r for r in ratings if 94 <= r < 98]),
                    'Excellent (90-93)': len([r for r in ratings if 90 <= r < 94]),
                    'Very Good (86-89)': len([r for r in ratings if 86 <= r < 90]),
                    'Good (80-85)': len([r for r in ratings if 80 <= r < 86]),
                    'Average (70-79)': len([r for r in ratings if 70 <= r < 80]),
                }

                categories = list(rating_categories.keys())
                counts = list(rating_categories.values())
                colors_map = [
                    'rgba(46, 125, 50, 0.85)',
                    'rgba(67, 160, 71, 0.85)',
                    'rgba(124, 179, 66, 0.85)',
                    'rgba(253, 216, 53, 0.85)',
                    'rgba(255, 179, 0, 0.85)',
                    'rgba(245, 124, 0, 0.85)'
                ]

                fig = go.Figure(data=[go.Bar(
                    y=categories,
                    x=counts,
                    orientation='h',
                    marker_color=colors_map,
                    text=counts,
                    textposition='auto'
                )])
                fig.update_layout(
                    xaxis_title="Wines",
                    yaxis_title="",
                    showlegend=False,
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, width="stretch")

    with col5:
        with st.container(border=True):
            st.markdown("#### Drinking Window Status")
            window_wines = drinking_window_wines

            import plotly.graph_objects as go

            ready_count = sum(w['bottles'] for w in window_wines['ready_now'])
            soon_count = sum(w['bottles'] for w in window_wines['drink_soon'])
            aging_count = sum(w['bottles'] for w in window_wines['for_aging'])

            labels = ['Ready Now', 'Drink Soon (1-2 yrs)', 'For Aging (3+ yrs)']
            values = [ready_count, soon_count, aging_count]
            colors = [
                'rgba(67, 160, 71, 0.85)',
                'rgba(255, 167, 38, 0.85)',
                'rgba(139, 26, 26, 0.85)'
            ]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo='label+percent',
                textposition='auto'
            )])
            fig.update_layout(
                showlegend=True,
                height=280,
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, width="stretch")

    with col6:
        with st.container(border=True):
            st.markdown("#### Wine Age Analysis")
            current_year = 2025

            age_ranges = {
                '0-5 years': 0,
                '6-10 years': 0,
                '11-15 years': 0,
                '16-20 years': 0,
                '20+ years': 0
            }

            for wine in inventory:
                vintage = wine.get('vintage')
                if vintage:
                    age = current_year - vintage
                    qty = wine.get('quantity', 0)
                    if age <= 5:
                        age_ranges['0-5 years'] += qty
                    elif age <= 10:
                        age_ranges['6-10 years'] += qty
                    elif age <= 15:
                        age_ranges['11-15 years'] += qty
                    elif age <= 20:
                        age_ranges['16-20 years'] += qty
                    else:
                        age_ranges['20+ years'] += qty

            if sum(age_ranges.values()) > 0:
                import plotly.graph_objects as go

                labels = list(age_ranges.keys())
                values = list(age_ranges.values())
                colors = [
                    'rgba(255, 224, 130, 0.85)',
                    'rgba(255, 183, 77, 0.85)',
                    'rgba(255, 152, 0, 0.85)',
                    'rgba(245, 124, 0, 0.85)',
                    'rgba(191, 54, 12, 0.85)'
                ]

                fig = go.Figure(data=[go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=values,
                    textposition='auto'
                )])
                fig.update_layout(
                    xaxis_title="Age Range",
                    yaxis_title="Bottles",
                    showlegend=False,
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    col7, col8, col9 = st.columns(3)

    # Row 3: Varietal/Grape Distribution
    with col7:
        with st.container(border=True):
            st.markdown("#### Top 5 Varietals")
            varietal_data = stats_repo.get_varietal_distribution(limit=5)

            if varietal_data:
                import plotly.graph_objects as go

                varietals = [v['varietal'] for v in varietal_data]
                bottles = [v['bottles'] for v in varietal_data]

                # Use solid purple color
                color = 'rgba(123, 31, 162, 0.85)'

                fig = go.Figure(data=[go.Bar(
                    y=varietals,
                    x=bottles,
                    orientation='h',
                    marker_color=color,
                    text=bottles,
                    textposition='auto'
                )])
                fig.update_layout(
                    xaxis_title="Bottles",
                    yaxis_title="",
                    showlegend=False,
                    height=320,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No varietal information available for wines in your cellar.")

    with col8:
        with st.container(border=True):
            st.markdown("#### Top 5 Regions")
            region_data = stats_repo.get_region_distribution(limit=5)

            if region_data:
                import plotly.graph_objects as go

                regions = [f"{r['region']}, {r['country']}" for r in region_data]
                bottles = [r['bottles'] for r in region_data]

                # Use solid green color (wine-growing regions)
                color = 'rgba(67, 160, 71, 0.85)'

                fig = go.Figure(data=[go.Bar(
                    y=regions,
                    x=bottles,
                    orientation='h',
                    marker_color=color,
                    text=bottles,
                    textposition='auto'
                )])
                fig.update_layout(
                    xaxis_title="Bottles",
                    yaxis_title="",
                    showlegend=False,
                    height=320,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No region information available for wines in your cellar.")

    with col9:
        with st.container(border=True):
            st.markdown("#### Cellar Size Over Time")
            size_data = stats_repo.get_cellar_size_over_time()

            if size_data:
                import plotly.graph_objects as go
                from datetime import datetime

                # Extract and format cellar-data for plotting
                months = []
                cumulative_bottles = []
                for data in size_data:
                    if data['month']:
                        # Format to show just YYYY-MM for better readability
                        month_display = data['month_display'] if data.get('month_display') else data['month'][:7]
                        months.append(month_display)
                        cumulative_bottles.append(data['cumulative_bottles'])

                # Use wine red color for the bars
                color = 'rgba(139, 69, 19, 0.85)'

                fig = go.Figure(data=go.Bar(
                    x=months,
                    y=cumulative_bottles,
                    marker_color=color,
                    text=cumulative_bottles,
                    textposition='auto',
                    name='Total Bottles'
                ))

                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Bottles",
                    showlegend=False,
                    height=320,
                    margin=dict(t=10, b=40, l=10, r=10),  # More bottom margin for rotated labels
                    xaxis=dict(
                        tickangle=45,
                        tickmode='array',
                        tickvals=months[::max(1, len(months)//6)],  # Show max 6 ticks to avoid crowding
                        ticktext=[m for m in months[::max(1, len(months)//6)]],
                        type='category'
                    )
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No CellarTracker bottle purchase cellar-data available for timeline chart.")

