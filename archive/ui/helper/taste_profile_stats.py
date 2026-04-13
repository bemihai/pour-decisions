"""Helper functions for Taste Profile page statistics and visualizations."""
import math
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.database import get_db_connection
from src.database.repository import StatsRepository, BottleRepository
from src.etl.utils import denormalize_rating, get_rating_description


def show_taste_profile_overview():
    """Display overview metrics for taste profile."""
    stats_repo = StatsRepository()

    # Get statistics
    rating_stats = stats_repo.get_rating_statistics()
    overall = rating_stats['overall']
    distribution = rating_stats.get('distribution', [])
    wine_type_stats = stats_repo.get_wine_type_stats()

    # Calculate metrics
    avg_rating = overall.get('avg_rating', 0)
    wines_rated = overall.get('wines_rated', 0)

    # Find favorite type (most consumed)
    favorite_type = "N/A"
    if wine_type_stats:
        favorite_type = wine_type_stats[0].get('wine_type', 'N/A')

    # Calculate percentage of highly-rated wines (90+)
    highly_rated_count = 0
    for d in distribution:
        if d.get('rating_range') == '90-100':
            highly_rated_count = d.get('count', 0)
            break

    highly_rated_pct = (highly_rated_count / wines_rated * 100) if wines_rated > 0 else 0

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Average Rating",
            value=f"{avg_rating:.1f}/100" if avg_rating else "N/A",
            delta=None
        )

    with col2:
        st.metric(
            label="Wines Tasted",
            value=wines_rated,
            delta=None
        )

    with col3:
        st.metric(
            label="Favorite Type",
            value=favorite_type,
            delta=None
        )

    with col4:
        st.metric(
            label="Highly Rated",
            value=f"{highly_rated_pct:.0f}%",
            delta=None,
            help=f"{highly_rated_count} wines rated 90+ out of {wines_rated} total"
        )


def show_rating_distribution():
    """Display rating distribution as donut chart with 5-point intervals."""
    stats_repo = StatsRepository()

    # Get consumed wines with ratings
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.personal_rating
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed' AND t.personal_rating IS NOT NULL
        """)
        ratings = [row['personal_rating'] for row in cursor.fetchall()]

    if not ratings:
        st.info("No rating data available yet.")
        return

    # Create 5-point intervals: 0-49, 50-54, 55-59, ..., 95-100
    ranges = []
    counts = []

    # 0-49 (poor wines)
    poor_count = sum(1 for r in ratings if r < 50)
    if poor_count > 0:
        ranges.append('0-49')
        counts.append(poor_count)

    # 50-94 in 5-point intervals
    for i in range(50, 95, 5):
        count = sum(1 for r in ratings if i <= r < i + 5)
        if count > 0:
            ranges.append(f'{i}-{i+4}')
            counts.append(count)

    # 95-100 (exceptional wines)
    excellent_count = sum(1 for r in ratings if r >= 95)
    if excellent_count > 0:
        ranges.append('95-100')
        counts.append(excellent_count)

    # Create color gradient from red to green
    num_ranges = len(ranges)
    if num_ranges <= 3:
        colors = ['#F44336', '#FFC107', '#4CAF50'][:num_ranges]
    else:
        # Create gradient for more ranges
        colors = []
        for i in range(num_ranges):
            ratio = i / (num_ranges - 1) if num_ranges > 1 else 0
            if ratio < 0.33:
                # Red to yellow
                colors.append(f'rgb({244}, {int(67 + (193 - 67) * (ratio / 0.33))}, {int(54 + (7 - 54) * (ratio / 0.33))})')
            elif ratio < 0.67:
                # Yellow to light green
                colors.append(f'rgb({int(255 - (255 - 139) * ((ratio - 0.33) / 0.34))}, {int(193 + (195 - 193) * ((ratio - 0.33) / 0.34))}, {int(7 + (74 - 7) * ((ratio - 0.33) / 0.34))})')
            else:
                # Light green to dark green
                colors.append(f'rgb({int(139 - (139 - 76) * ((ratio - 0.67) / 0.33))}, {int(195 + (175 - 195) * ((ratio - 0.67) / 0.33))}, {int(74 + (80 - 74) * ((ratio - 0.67) / 0.33))})')

    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=ranges,
        values=counts,
        hole=0.4,
        marker=dict(colors=colors)
    )])

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )

    st.plotly_chart(fig, width="stretch")


def show_wine_type_distribution():
    """Display wine type distribution as donut chart."""
    stats_repo = StatsRepository()
    wine_type_stats = stats_repo.get_wine_type_stats()

    if not wine_type_stats:
        st.info("No wine type data available yet.")
        return

    st.markdown("### <i class='fa-solid fa-wine-glass fa-icon'></i>Wine Type Distribution", unsafe_allow_html=True)

    # Donut chart
    types = [w['wine_type'] for w in wine_type_stats]
    counts = [w['wines_tasted'] for w in wine_type_stats]

    fig = go.Figure(data=[go.Pie(
        labels=types,
        values=counts,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )

    st.plotly_chart(fig, width="stretch")


def show_wine_type_performance():
    """Display wine type performance table."""
    stats_repo = StatsRepository()
    wine_type_stats = stats_repo.get_wine_type_stats()

    if not wine_type_stats:
        st.info("No wine type data available yet.")
        return

    st.markdown("### <i class='fa-solid fa-star fa-icon'></i>Performance by Type", unsafe_allow_html=True)

    for wine_type_data in wine_type_stats:
        wine_type = wine_type_data.get('wine_type', 'Unknown')
        wines_tasted = wine_type_data.get('wines_tasted', 0)
        avg_rating = wine_type_data.get('avg_rating')
        highest = wine_type_data.get('highest_rating')

        with st.expander(f"{wine_type} ({wines_tasted} wine{'s' if wines_tasted != 1 else ''})"):
            if avg_rating:
                st.write(f"**Average Rating:** {avg_rating:.1f}/100")
                st.write(f"**Highest Rating:** {highest:.0f}/100")
            else:
                st.write("**Average Rating:** Not rated")


def show_top_varietals():
    """Display top 3 varietal preferences as cards."""
    stats_repo = StatsRepository()
    varietals = stats_repo.get_varietal_preferences(limit=10)

    if not varietals:
        st.info("No varietal data available yet.")
        return

    st.markdown("### <i class='fa-solid fa-seedling fa-icon'></i>Top 3 Varietals", unsafe_allow_html=True)

    # Show top 3 as cards
    top_3 = varietals[:3]
    if top_3:
        cols = st.columns(3)
        for idx, (col, varietal_data) in enumerate(zip(cols, top_3), 1):
            with col:
                varietal = varietal_data.get('varietal', 'Unknown')
                count = varietal_data.get('wines_tasted', 0)
                avg_rating = varietal_data.get('avg_rating')

                st.markdown(f"**#{idx} {varietal}**")
                st.write(f"🍷 {count} wine{'s' if count != 1 else ''}")
                if avg_rating:
                    # Create star display
                    denorm = denormalize_rating(avg_rating)
                    stars = "⭐" * int(denorm) if denorm else ""
                    st.write(f"⭐ {avg_rating:.1f}/100")
                st.markdown("---")
    else:
        st.info("Not enough varietal data to display top 3.")


def show_varietal_analysis():
    """Display varietal analysis chart with all top varietals."""
    stats_repo = StatsRepository()
    varietals = stats_repo.get_varietal_preferences(limit=10)

    if not varietals:
        st.info("No varietal data available yet.")
        return

    st.markdown("### <i class='fa-solid fa-chart-line fa-icon'></i>Varietal Analysis", unsafe_allow_html=True)

    names = [v['varietal'] for v in varietals]
    counts = [v['wines_tasted'] for v in varietals]
    ratings = [v.get('avg_rating', 0) for v in varietals]

    fig = go.Figure()

    # Add bar for count
    fig.add_trace(go.Bar(
        name='Wines Tasted',
        x=names,
        y=counts,
        marker_color='rgba(123, 31, 162, 0.7)',
        yaxis='y',
        offsetgroup=1
    ))

    # Add line for average rating
    fig.add_trace(go.Scatter(
        name='Avg Rating',
        x=names,
        y=ratings,
        mode='lines+markers',
        marker=dict(color='#FFC107', size=8),
        line=dict(color='#FFC107', width=2),
        yaxis='y2'
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=80),
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="Wines Tasted", side='left'),
        yaxis2=dict(title="Average Rating", side='right', overlaying='y', range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )

    st.plotly_chart(fig, width="stretch")


def show_producer_loyalty():
    """Display favorite producers."""
    stats_repo = StatsRepository()
    producers = stats_repo.get_producer_preferences(limit=5)

    if not producers:
        st.info("No producer data available yet.")
        return

    st.markdown("### <i class='fa-solid fa-building fa-icon'></i>Favorite Producers", unsafe_allow_html=True)

    for idx, producer_data in enumerate(producers, 1):
        producer = producer_data.get('producer_name', 'Unknown')
        country = producer_data.get('country', 'Unknown')
        wines_tasted = producer_data.get('wines_tasted', 0)
        avg_rating = producer_data.get('avg_rating')
        highest = producer_data.get('highest_rating')

        with st.expander(f"#{idx} {producer} ({country}) - {wines_tasted} wine{'s' if wines_tasted != 1 else ''}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Producer:** {producer}")
                st.write(f"**Country:** {country}")
                st.write(f"**Wines Tasted:** {wines_tasted}")

            with col2:
                if avg_rating:
                    st.write(f"**Average Rating:** {avg_rating:.1f}/100")
                    st.write(f"**Highest Rating:** {highest:.0f}/100")

                    # Show stars for average
                    denorm = denormalize_rating(avg_rating)
                    if denorm:
                        stars_html = "⭐" * int(denorm)
                        st.markdown(f"{stars_html}")


def show_favorite_regions():
    """Display favorite regions."""
    stats_repo = StatsRepository()
    regions = stats_repo.get_region_preferences(limit=5)

    if not regions:
        st.info("No region data available yet.")
        return

    st.markdown("### <i class='fa-solid fa-map-location-dot fa-icon'></i>Favorite Regions", unsafe_allow_html=True)

    for idx, region_data in enumerate(regions, 1):
        region = region_data.get('region_name', 'Unknown')
        country = region_data.get('country', 'Unknown')
        wines_tasted = region_data.get('wines_tasted', 0)
        avg_rating = region_data.get('avg_rating')
        highest = region_data.get('highest_rating')

        with st.expander(f"#{idx} {region} ({country}) - {wines_tasted} wine{'s' if wines_tasted != 1 else ''}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Region:** {region}")
                st.write(f"**Country:** {country}")
                st.write(f"**Wines Tasted:** {wines_tasted}")

            with col2:
                if avg_rating:
                    st.write(f"**Average Rating:** {avg_rating:.1f}/100")
                    st.write(f"**Highest Rating:** {highest:.0f}/100")

                    # Show stars for average
                    denorm = denormalize_rating(avg_rating)
                    if denorm:
                        stars_html = "⭐" * int(denorm)
                        st.markdown(f"{stars_html}")


def show_rating_trends():
    """Display rating trends over time."""
    stats_repo = StatsRepository()
    timeline = stats_repo.get_rating_timeline()

    if not timeline or len(timeline) < 2:
        st.info("Not enough data to show rating trends. Keep tasting wines!")
        return

    # Filter to show only the most recent 12 months
    timeline_recent = timeline[-12:] if len(timeline) > 12 else timeline

    # Prepare data
    months = [t['month'] for t in timeline_recent]
    ratings = [t['avg_rating'] for t in timeline_recent]
    counts = [t['wines_count'] for t in timeline_recent]

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add rating line
    fig.add_trace(go.Scatter(
        name='Average Rating',
        x=months,
        y=ratings,
        mode='lines+markers',
        marker=dict(color='#7b1fa2', size=8),
        line=dict(color='#7b1fa2', width=3),
        yaxis='y'
    ))

    # Add count bars
    fig.add_trace(go.Bar(
        name='Wines Tasted',
        x=months,
        y=counts,
        marker_color='rgba(123, 31, 162, 0.3)',
        yaxis='y2'
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Average Rating", side='left', range=[0, 100]),
        yaxis2=dict(title="Wines Tasted", side='right', overlaying='y'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )

    st.plotly_chart(fig, width="stretch")

    # Show trend insight
    if len(ratings) >= 2:
        trend = "improving" if ratings[-1] > ratings[0] else "declining" if ratings[-1] < ratings[0] else "stable"
        trend_color = "🟢" if trend == "improving" else "🔴" if trend == "declining" else "🟡"
        st.caption(f"{trend_color} Your ratings are {trend} over time (from {ratings[0]:.1f} to {ratings[-1]:.1f})")


def show_consumed_wines_inventory():
    """Display consumed wines with filtering options."""
    bottle_repo = BottleRepository()

    # Get all consumed wines
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                b.*,
                w.wine_name, w.wine_type, w.vintage, w.varietal,
                p.name as producer_name,
                r.country, 
                COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                t.personal_rating, t.community_rating, t.tasting_notes, t.last_tasted_date
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN producers p ON w.producer_id = p.id
            LEFT JOIN regions r ON w.region_id = r.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed'
            ORDER BY b.consumed_date DESC
        """)
        all_consumed = [dict(row) for row in cursor.fetchall()]

    if not all_consumed:
        st.info("No consumed wines found yet.")
        return

    # Extract unique values for filters
    wine_types = sorted(set(w.get('wine_type') for w in all_consumed if w.get('wine_type')))
    countries = sorted(set(w.get('country') for w in all_consumed if w.get('country')))
    producers = sorted(set(w.get('producer_name') for w in all_consumed if w.get('producer_name')))

    # Get vintage range
    vintages = [w.get('vintage') for w in all_consumed if w.get('vintage')]
    min_vintage = min(vintages) if vintages else 2000
    max_vintage = max(vintages) if vintages else 2024

    # Create filter UI
    with st.container(border=True):
        st.markdown("### <i class='fa-solid fa-filter fa-icon'></i>Filter Consumed Wines", unsafe_allow_html=True)
        st.markdown("")

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            wine_types_with_all = ["All Types"] + wine_types
            selected_type = st.selectbox("Wine Type", wine_types_with_all)

        with filter_col2:
            countries_with_all = ["All Countries"] + countries
            selected_country = st.selectbox("Country", countries_with_all)

        with filter_col3:
            producers_with_all = ["All Producers"] + producers
            selected_producer = st.selectbox("Producer", producers_with_all)

        with filter_col4:
            rating_filter = st.selectbox("Rating", ["All Ratings", "Rated Only", "Unrated", "90+", "80+", "70+"])

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
            search_term = st.text_input("Search", placeholder="Wine name, varietal...")

        with filter_col7:
            sort_by = st.selectbox("Sort By", [
                "Consumed Date (Recent→Old)",
                "Consumed Date (Old→Recent)",
                "Rating (High→Low)",
                "Rating (Low→High)",
                "Producer",
                "Wine Name"
            ])

        with filter_col8:
            limit = st.number_input("Limit Results", min_value=5, max_value=100, value=20, step=5)

    # Apply filters
    filtered_consumed = all_consumed

    # Filter by wine type
    if selected_type != "All Types":
        filtered_consumed = [w for w in filtered_consumed if w.get('wine_type') == selected_type]

    # Filter by country
    if selected_country != "All Countries":
        filtered_consumed = [w for w in filtered_consumed if w.get('country') == selected_country]

    # Filter by producer
    if selected_producer != "All Producers":
        filtered_consumed = [w for w in filtered_consumed if w.get('producer_name') == selected_producer]

    # Filter by vintage range
    filtered_consumed = [
        w for w in filtered_consumed
        if w.get('vintage') is None or (vintage_range[0] <= w.get('vintage') <= vintage_range[1])
    ]

    # Filter by rating
    if rating_filter == "Rated Only":
        filtered_consumed = [w for w in filtered_consumed if w.get('personal_rating') is not None]
    elif rating_filter == "Unrated":
        filtered_consumed = [w for w in filtered_consumed if w.get('personal_rating') is None]
    elif rating_filter == "90+":
        filtered_consumed = [w for w in filtered_consumed if w.get('personal_rating', 0) >= 90]
    elif rating_filter == "80+":
        filtered_consumed = [w for w in filtered_consumed if w.get('personal_rating', 0) >= 80]
    elif rating_filter == "70+":
        filtered_consumed = [w for w in filtered_consumed if w.get('personal_rating', 0) >= 70]

    # Filter by search term
    if search_term:
        search_lower = search_term.lower()
        filtered_consumed = [
            w for w in filtered_consumed
            if search_lower in w.get('wine_name', '').lower()
            or search_lower in w.get('producer_name', '').lower()
            or search_lower in (w.get('varietal', '') or '').lower()
        ]

    # Sort
    if sort_by == "Consumed Date (Recent→Old)":
        filtered_consumed.sort(key=lambda w: w.get('consumed_date') or '0000-00-00', reverse=True)
    elif sort_by == "Consumed Date (Old→Recent)":
        filtered_consumed.sort(key=lambda w: w.get('consumed_date') or '9999-99-99')
    elif sort_by == "Rating (High→Low)":
        filtered_consumed.sort(key=lambda w: w.get('personal_rating') or 0, reverse=True)
    elif sort_by == "Rating (Low→High)":
        filtered_consumed.sort(key=lambda w: w.get('personal_rating') or 9999)
    elif sort_by == "Producer":
        filtered_consumed.sort(key=lambda w: (w.get('producer_name', ''), w.get('vintage') or 0))
    elif sort_by == "Wine Name":
        filtered_consumed.sort(key=lambda w: w.get('wine_name', ''))

    # Apply limit
    filtered_consumed = filtered_consumed[:limit]

    if not filtered_consumed:
        st.warning("No wines found matching the selected filters.")
        return

    # Results header
    st.markdown("")
    st.markdown(f"### Consumed Wines ({len(filtered_consumed)} wines)", unsafe_allow_html=True)
    st.markdown("")

    # Display wines in expandable sections
    for wine_data in filtered_consumed:
        wine_name = wine_data.get('wine_name', 'Unknown')
        producer_name = wine_data.get('producer_name', 'Unknown Producer')
        vintage = wine_data.get('vintage')
        wine_type = wine_data.get('wine_type', 'Unknown')
        country = wine_data.get('country', 'Unknown')
        region_name = wine_data.get('region_name', '')
        consumed_date = wine_data.get('consumed_date', '')
        rating = wine_data.get('personal_rating')
        tasting_notes = wine_data.get('tasting_notes', '')

        # Create title with rating if available
        title_parts = [f"{producer_name}, {wine_name} ({vintage or 'NV'})"]
        if rating:
            title_parts.append(f"- {rating}/100")
        if consumed_date:
            title_parts.append(f"- Consumed: {consumed_date}")

        with st.expander(" ".join(title_parts)):
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
                st.write("**Consumption Info**")
                if consumed_date:
                    st.write(f"Consumed: {consumed_date}")
                else:
                    st.write("Consumed: Date unknown")

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

            if tasting_notes:
                st.markdown("---")
                st.write("**Tasting Notes:**")
                st.write(tasting_notes)


def show_favorite_countries():
    """Display favorite countries based on consumed wines."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                r.country,
                COUNT(DISTINCT b.id) as wines_tasted,
                AVG(t.personal_rating) as avg_rating,
                MAX(t.personal_rating) as highest_rating
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN regions r ON w.region_id = r.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed' AND r.country IS NOT NULL
            GROUP BY r.country
            HAVING COUNT(DISTINCT b.id) >= 1
            ORDER BY wines_tasted DESC, avg_rating DESC
            LIMIT 5
        """)
        countries = [dict(row) for row in cursor.fetchall()]

    if not countries:
        st.info("No country data available yet.")
        return

    st.markdown("### <i class='fa-solid fa-globe fa-icon'></i>Favorite Countries", unsafe_allow_html=True)

    for idx, country_data in enumerate(countries, 1):
        country = country_data.get('country', 'Unknown')
        wines_tasted = country_data.get('wines_tasted', 0)
        avg_rating = country_data.get('avg_rating')
        highest = country_data.get('highest_rating')

        with st.expander(f"#{idx} {country} - {wines_tasted} wine{'s' if wines_tasted != 1 else ''}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Country:** {country}")
                st.write(f"**Wines Tasted:** {wines_tasted}")

            with col2:
                if avg_rating:
                    st.write(f"**Average Rating:** {avg_rating:.1f}/100")
                    st.write(f"**Highest Rating:** {highest:.0f}/100")

                    denorm = denormalize_rating(avg_rating)
                    if denorm:
                        stars_html = "⭐" * int(denorm)
                        st.markdown(f"{stars_html}")


def show_favorite_vintages():
    """Display favorite vintages based on consumed wines."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                w.vintage,
                COUNT(DISTINCT b.id) as wines_tasted,
                AVG(t.personal_rating) as avg_rating,
                MAX(t.personal_rating) as highest_rating
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed' AND w.vintage IS NOT NULL
            GROUP BY w.vintage
            HAVING COUNT(DISTINCT b.id) >= 2
            ORDER BY avg_rating DESC, wines_tasted DESC
            LIMIT 5
        """)
        vintages = [dict(row) for row in cursor.fetchall()]

    if not vintages:
        st.info("No vintage data available yet.")
        return

    st.markdown("### <i class='fa-solid fa-calendar fa-icon'></i>Top Vintages", unsafe_allow_html=True)

    for idx, vintage_data in enumerate(vintages, 1):
        vintage = vintage_data.get('vintage', 'Unknown')
        wines_tasted = vintage_data.get('wines_tasted', 0)
        avg_rating = vintage_data.get('avg_rating')
        highest = vintage_data.get('highest_rating')

        with st.expander(f"#{idx} {vintage} - {wines_tasted} wine{'s' if wines_tasted != 1 else ''}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Vintage:** {vintage}")
                st.write(f"**Wines Tasted:** {wines_tasted}")

            with col2:
                if avg_rating:
                    st.write(f"**Average Rating:** {avg_rating:.1f}/100")
                    st.write(f"**Highest Rating:** {highest:.0f}/100")

                    denorm = denormalize_rating(avg_rating)
                    if denorm:
                        stars_html = "⭐" * int(denorm)
                        st.markdown(f"{stars_html}")


def show_favorite_appellations():
    """Display favorite appellations based on consumed wines."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                w.appellation,
                r.country,
                COUNT(DISTINCT b.id) as wines_tasted,
                AVG(t.personal_rating) as avg_rating,
                MAX(t.personal_rating) as highest_rating
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN regions r ON w.region_id = r.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed' AND w.appellation IS NOT NULL
            GROUP BY w.appellation
            HAVING COUNT(DISTINCT b.id) >= 1
            ORDER BY wines_tasted DESC, avg_rating DESC
            LIMIT 5
        """)
        appellations = [dict(row) for row in cursor.fetchall()]

    if not appellations:
        st.info("No appellation data available yet.")
        return

    st.markdown("### <i class='fa-solid fa-award fa-icon'></i>Favorite Appellations", unsafe_allow_html=True)

    for idx, app_data in enumerate(appellations, 1):
        appellation = app_data.get('appellation', 'Unknown')
        country = app_data.get('country', 'Unknown')
        wines_tasted = app_data.get('wines_tasted', 0)
        avg_rating = app_data.get('avg_rating')
        highest = app_data.get('highest_rating')

        with st.expander(f"#{idx} {appellation} ({country}) - {wines_tasted} wine{'s' if wines_tasted != 1 else ''}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Appellation:** {appellation}")
                st.write(f"**Country:** {country}")
                st.write(f"**Wines Tasted:** {wines_tasted}")

            with col2:
                if avg_rating:
                    st.write(f"**Average Rating:** {avg_rating:.1f}/100")
                    st.write(f"**Highest Rating:** {highest:.0f}/100")

                    denorm = denormalize_rating(avg_rating)
                    if denorm:
                        stars_html = "⭐" * int(denorm)
                        st.markdown(f"{stars_html}")




