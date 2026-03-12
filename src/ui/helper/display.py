"""Custom CSS/HTML code used in the UI."""
import streamlit as st
import html


def get_drinking_status(drink_index: float, all_indices: list[float]) -> tuple[str, str]:
    """
    Derive drinking status label and colour for a given drink_index.

    Args:
        drink_index: The drinking index value for the current wine
        all_indices: List of all drinking index values in the collection for normalisation

    Returns:
        Tuple of (status_label, hex_colour)
    """
    if not all_indices:
        normalized = 50.0
    else:
        sorted_idx = sorted(all_indices)
        p5  = sorted_idx[max(0, int(len(sorted_idx) * 0.05))]
        p95 = sorted_idx[min(len(sorted_idx) - 1, int(len(sorted_idx) * 0.95))]
        lo, hi = (p5, p95) if p5 != p95 else (sorted_idx[0], sorted_idx[-1])
        if hi == lo:
            normalized = 50.0
        else:
            normalized = max(0.0, min(100.0, ((drink_index - lo) / (hi - lo)) * 100))

    if normalized >= 75:
        return "Peak Drinking", "#4CAF50"
    elif normalized >= 50:
        return "Ready to Drink", "#FFC107"
    elif normalized >= 25:
        return "Approaching", "#FF9800"
    else:
        return "Hold", "#F44336"


def render_drinking_index_bar(drink_index: float, all_indices: list[float]) -> None:
    """
    Render a visual progress bar for drinking index.

    Args:
        drink_index: The drinking index value for the current wine
        all_indices: List of all drinking index values in the collection for normalization
    """
    if not all_indices:
        return

    status, color = get_drinking_status(drink_index, all_indices)

    sorted_idx = sorted(all_indices)
    p5  = sorted_idx[max(0, int(len(sorted_idx) * 0.05))]
    p95 = sorted_idx[min(len(sorted_idx) - 1, int(len(sorted_idx) * 0.95))]
    lo, hi = (p5, p95) if p5 != p95 else (sorted_idx[0], sorted_idx[-1])
    normalized = max(0.0, min(100.0, ((drink_index - lo) / (hi - lo) * 100) if hi != lo else 50))

    bar_text = "Drink Sooner" if normalized >= 50 else "Drink Later"

    st.write(f"Status: {status}")

    # Create visual progress bar with text (60% width for more compact display)
    st.markdown(f"""
    <div style="background-color: #e0e0e0; border-radius: 10px; height: 22px; width: 60%; position: relative; margin-top: 5px;">
        <div style="background-color: {color}; border-radius: 10px; height: 22px; width: {normalized}%; position: absolute; top: 0; left: 0;"></div>
        <div style="position: absolute; top: 0; left: 0; width: 100%; text-align: center; line-height: 22px; font-size: 12px; font-weight: bold; color: #333;">
            {bar_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_community_cellar_bar(q_purchased: int, q_consumed: int, q_quantity: int) -> None:
    """
    Render a visual stacked bar showing community cellar stats as percentages.

    Displays what proportion of community-purchased bottles are currently held in
    cellars vs consumed, using the same progress-bar style as the drinking index bar.

    Args:
        q_purchased: Total bottles purchased by the community
        q_consumed: Total bottles consumed by the community
        q_quantity: Total bottles currently held in community cellars
    """
    if q_purchased <= 0:
        return

    pct_held = min((q_quantity / q_purchased) * 100, 100)
    pct_consumed = min((q_consumed / q_purchased) * 100, 100)

    st.markdown(f"""
    <div style="background-color: #e0e0e0; border-radius: 10px; height: 22px; width: 60%; position: relative; margin-top: 5px; overflow: hidden;">
        <div style="background-color: #ce93d8; height: 22px; width: {pct_held:.1f}%; position: absolute; top: 0; left: 0;"></div>
        <div style="background-color: #a5d6a7; height: 22px; width: {pct_consumed:.1f}%; position: absolute; top: 0; left: {pct_held:.1f}%;"></div>
        <div style="position: absolute; top: 0; left: 0; width: 100%; text-align: center; line-height: 22px; font-size: 12px; font-weight: bold; color: #333;">
            {pct_held:.0f}% held &nbsp;|&nbsp; {pct_consumed:.0f}% consumed
        </div>
    </div>
    <div style="margin-top: 4px; font-size: 11px; color: #666; width: 60%;">
        <span style="color: #ce93d8; font-weight: bold;">&#9632;</span> In cellars &nbsp;
        <span style="color: #a5d6a7; font-weight: bold;">&#9632;</span> Consumed
    </div>
    """, unsafe_allow_html=True)


def render_community_rating(community_rating: float | None, like_percentage: float | None, like_votes: int | None) -> None:
    """
    Render community rating and like percentage as visual progress bars.

    Uses the same progress-bar style as the drinking index and community cellar bars.

    Args:
        community_rating: Community average rating on 0-100 scale
        like_percentage: Percentage of community 'like' votes as a fraction (0.0 to 1.0)
        like_votes: Total number of community 'like' votes
    """
    if community_rating is None and like_percentage is None:
        return

    bars_html = ""

    if community_rating is not None:
        bars_html += f"""
    <div style="margin-top: 8px;">
        <div style="font-size: 12px; color: #555; margin-bottom: 3px;">Community Rating: {community_rating:.0f}/100</div>
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 22px; width: 60%; position: relative; overflow: hidden;">
            <div style="background-color: #ffcc80; height: 22px; width: {community_rating:.1f}%; position: absolute; top: 0; left: 0;"></div>
            <div style="position: absolute; top: 0; left: 0; width: 100%; text-align: center; line-height: 22px; font-size: 12px; font-weight: bold; color: #333;">
                {community_rating:.0f}/100
            </div>
        </div>
    </div>"""

    if like_percentage is not None:
        pct = like_percentage * 100
        votes_text = f" ({like_votes:,} votes)" if like_votes else ""
        bars_html += f"""
    <div style="margin-top: 6px;">
        <div style="font-size: 12px; color: #555; margin-bottom: 3px;">Liked by{votes_text}: {pct:.0f}%</div>
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 22px; width: 60%; position: relative; overflow: hidden;">
            <div style="background-color: #80cbc4; height: 22px; width: {pct:.1f}%; position: absolute; top: 0; left: 0;"></div>
            <div style="position: absolute; top: 0; left: 0; width: 100%; text-align: center; line-height: 22px; font-size: 12px; font-weight: bold; color: #333;">
                {pct:.0f}%
            </div>
        </div>
    </div>"""

    st.markdown(bars_html, unsafe_allow_html=True)


TABS_DISPLAY = """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
            padding: 10px 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: white;
            border-radius: 8px;
            padding: 0 24px;
            font-weight: 500;
            border: 1px solid #e0e0e0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #7b1fa2;
            color: white;
            border-color: #7b1fa2;
        }
        /* Better vertical spacing */
        div[data-testid="stVerticalBlock"] > div {
            gap: 1.5rem;
        }
        /* Font Awesome icon styling */
        .fa-icon {
            margin-right: 6px;
            color: #7b1fa2;
        }
        /* Compact generate/regenerate description buttons inside expanders */
        div[data-testid="stExpander"] button[kind="secondary"] {
            font-size: 11px !important;
            padding: 2px 8px !important;
            height: auto !important;
            min-height: 0 !important;
            line-height: 1.4 !important;
        }
        div[data-testid="stExpander"] button[kind="secondary"] p {
            font-size: 11px !important;
        }
        </style>
    """


CONTENT_STYLE = """
<style> 
#input-container { 
    position: fixed; 
    bottom: 0; 
    width: 100%; 
    padding: 10px; 
    background-color: white; 
    z-index: 100; 
}

.user-avatar, .bot-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    margin-bottom: -10px;
}
.user-avatar {
    float: right;
    margin-left: 5px;
}
.bot-avatar {
    float: left;
    margin-right: 5px;
}
.user-bubble {
    background: #e6f4ea;
    color: #222;
    border-radius: 20px 20px 4px 20px;
    padding: 14px 18px;
    margin-right: 8px;
    max-width: 70%;
    font-size: 0.98em;
    box-shadow: 0 2px 8px rgba(91,140,42,0.08);
    line-height: 1.5;
    word-break: break-word;
}
.bot-bubble {
    background: #ede7f6;
    color: #222;
    border-radius: 20px 20px 20px 4px;
    padding: 14px 18px;
    max-width: 70%;
    font-size: 0.98em;
    box-shadow: 0 2px 8px rgba(123,31,162,0.08);
    line-height: 1.5;
    word-break: break-word;
}
.sources-container {
    margin-top: 12px;
    padding: 8px 0;
    font-size: 0.65em;
}
.sources-title {
    font-weight: 600;
    color: #7b1fa2;
    margin-bottom: 4px;
    font-size: 0.85em;
}
.source-item {
    padding: 2px 0;
    color: #555;
    line-height: 1.2;
}
.source-name {
    font-weight: 500;
    color: #333;
}
.source-details {
    color: #666;
    font-size: 0.85em;
}
</style>
"""


def make_page_title(title: str, subtitle: str) -> str:
    """
    Returns a styled HTML/CSS string for a custom page title and subtitle (centered).
    Args:
        title (str): The main title text.
        subtitle (str): The subtitle text (smaller font).
    Returns:
        str: HTML/CSS for the styled title and subtitle.
    """
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;900&display=swap');
    .rag-title {{
      font-family: 'Poppins', sans-serif;
      font-weight: 700;
      font-size: 4em;
      background: linear-gradient(90deg, #7b1fa2 0%, #a4508b 50%, #5e3370 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      color: transparent;
      margin: 0;
      padding: 20px 0 0 0;
      text-align: center;
    }}
    .rag-subtitle {{
      font-family: 'Poppins', sans-serif;
      font-weight: 400;
      font-size: 1.5em;
      color: #5b8c2a;
      margin: 0;
      padding: 0 0 20px 0;
      text-align: center;
    }}
    </style>
    <div class="rag-title">{title}</div>
    <div class="rag-subtitle">{subtitle}</div>
    """


def make_compact_page_title(title: str, subtitle: str) -> str:
    """
    Returns a styled HTML/CSS string for a compact page title and subtitle (left-aligned).
    Specifically designed for pages that need more space for content.

    Args:
        title (str): The main title text.
        subtitle (str): The subtitle text (smaller font).
    Returns:
        str: HTML/CSS for the styled title and subtitle.
    """
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;900&display=swap');
    .compact-title {{
      font-family: 'Poppins', sans-serif;
      font-weight: 700;
      font-size: 2.5em;
      background: linear-gradient(90deg, #7b1fa2 0%, #a4508b 50%, #5e3370 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      color: transparent;
      margin: 0;
      padding: 10px 0 5px 0;
      text-align: left;
    }}
    .compact-subtitle {{
      font-family: 'Poppins', sans-serif;
      font-weight: 400;
      font-size: 1.1em;
      color: #5b8c2a;
      margin: 0;
      padding: 0 0 10px 0;
      text-align: left;
    }}
    </style>
    <div class="compact-title">{title}</div>
    <div class="compact-subtitle">{subtitle}</div>
    """


def get_relevance_indicator(score: float) -> str:
    """
    Get a visual indicator for source relevance quality.

    Args:
        score: Relevance score (0.0 to 1.0)

    Returns:
        Emoji indicator for quality level
    """
    if score >= 0.8:
        return "🟢"  # Green circle - Excellent
    elif score >= 0.6:
        return "🟡"  # Yellow circle - Good
    elif score >= 0.4:
        return "🟠"  # Orange circle - Fair
    else:
        return "🔴"  # Red circle - Low


def format_user_message(message: dict) -> str:
    """Format a user message with improved style and emoji avatar"""
    message_text = html.escape(message["question"])
    avatar_emoji = "🧑‍💼"  # person in suit
    return f"""
    <div style="display:flex; align-items:flex-end; justify-content:flex-end; margin-bottom:18px;">
        <div class="user-bubble">{message_text}</div>
        <span class="user-avatar" style="display:flex; align-items:center; justify-content:center; font-size:2em; background:#e6f4ea;">{avatar_emoji}</span>
    </div>
    """


def format_assistant_message(message: dict, sources: list = None, web_sources: list = None) -> str:
    """Format an assistant message with improved style and emoji avatar.

    Args:
        message: Dictionary containing the answer text.
        sources: Optional list of tuples (source_name, page_number, relevance_score).
        web_sources: Optional list of dicts with 'title' and 'url' keys from web search.

    Returns:
        HTML string with formatted message and sources.
    """
    # Handle case where answer might be a list or other non-string type
    try:
        answer = message["answer"]
        if isinstance(answer, list):
            # Extract text from content blocks
            text_parts = []
            for item in answer:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    text_parts.append(str(item))
            answer = " ".join(text_parts) if text_parts else ""
        elif not isinstance(answer, str):
            answer = str(answer)

        message_text = html.escape(answer)
    except Exception as e:
        # Fallback if there's any issue processing the answer
        message_text = html.escape("Error displaying message: " + str(e))
    avatar_emoji = "🍇" # grapes

    # Build sources HTML if provided
    sources_html = ""
    if sources:
        # Group sources by name and keep track of indexes, pages and highest relevance
        grouped_sources = {}
        for idx, (source_name, page_number, relevance_score) in enumerate(sources, 1):
            if source_name not in grouped_sources:
                grouped_sources[source_name] = {
                    'indexes': [],
                    'pages': [],
                    'max_relevance': relevance_score
                }

            # Add original index
            grouped_sources[source_name]['indexes'].append(idx)

            # Add page if valid
            if page_number is not None and page_number >= 0:
                if page_number not in grouped_sources[source_name]['pages']:
                    grouped_sources[source_name]['pages'].append(page_number)

            # Update max relevance
            if relevance_score is not None:
                current_max = grouped_sources[source_name]['max_relevance']
                if current_max is None or relevance_score > current_max:
                    grouped_sources[source_name]['max_relevance'] = relevance_score

        # Build HTML for grouped sources
        sources_items = []
        for source_name, data in grouped_sources.items():
            indexes = data['indexes']
            pages = sorted(data['pages'])
            relevance_score = data['max_relevance']

            # Format indexes
            if len(indexes) == 1:
                indexes_text = f"{indexes[0]}."
            else:
                indexes_text = f"{', '.join(map(str, indexes))}."

            # Format pages
            if pages:
                if len(pages) == 1:
                    page_text = f", Page {pages[0]}"
                elif len(pages) <= 3:
                    page_text = f", Pages {', '.join(map(str, pages))}"
                else:
                    page_text = f", Pages {pages[0]}-{pages[-1]}"
            else:
                page_text = ""

            # Add visual indicator at the front
            if relevance_score is not None:
                indicator = get_relevance_indicator(relevance_score)
            else:
                indicator = "⚪"  # White circle for unknown relevance

            sources_items.append(
                f'<div class="source-item">'
                f'{indicator} <span class="source-name">{indexes_text} {html.escape(str(source_name))}</span>'
                f'<span class="source-details">{page_text}</span>'
                f'</div>'
            )

        sources_html = f"""
<div class="sources-container">
    <div class="sources-title">📚 Sources:</div>
    {''.join(sources_items)}
</div>"""

    # Build web sources HTML if provided
    web_sources_html = ""
    if web_sources:
        web_items = []
        for source in web_sources:
            url = html.escape(source.get("url", ""))
            title = html.escape(source.get("title", url))
            web_items.append(
                f'<div class="source-item">'
                f'🌐 <a class="source-name" href="{url}" target="_blank">{title}</a>'
                f'</div>'
            )
        web_sources_html = f"""
<div class="sources-container">
    <div class="sources-title">🔍 Web Sources:</div>
    {''.join(web_items)}
</div>"""

    return f"""
<div style="display:flex; align-items:flex-start; justify-content:flex-start; margin-bottom:18px;">
    <span class="bot-avatar" style="display:flex; align-items:center; justify-content:center; font-size:2em; background:#ede7f6;">{avatar_emoji}</span>
    <div class="bot-bubble">
        {message_text}{sources_html}{web_sources_html}
    </div>
</div>"""


def display_message(message: dict):
    """Display a message in the UI.

    Args:
        message: Dictionary containing role, question/answer, and optionally sources.
                 For AI messages, may include 'sources' as a list of tuples
                 (source_name, page_number, relevance_score).
    """
    if message["role"] == "human":
        container_html = format_user_message(message)
        st.markdown(container_html, unsafe_allow_html=True)

    if message["role"] == "ai":
        sources = message.get("sources", [])
        web_sources = message.get("web_sources", [])
        container_html = format_assistant_message(message, sources, web_sources)
        st.markdown(container_html, unsafe_allow_html=True)
