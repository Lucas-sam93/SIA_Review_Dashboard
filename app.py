from pathlib import Path

import pandas as pd
import streamlit as st
import subprocess
import sys
from collections import Counter
from html import escape
import re
from urllib.parse import quote


st.set_page_config(page_title="SIA Review Pulse", layout="wide")


def ensure_wordcloud():
    try:
        from wordcloud import WordCloud, STOPWORDS  # type: ignore
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])
        from wordcloud import WordCloud, STOPWORDS  # type: ignore

    return WordCloud, STOPWORDS


def build_keyword_cloud(text_series: pd.Series):
    WordCloud, STOPWORDS = ensure_wordcloud()
    custom_stopwords = STOPWORDS.union(
        {
            "airline",
            "airlines",
            "flight",
            "flights",
            "seat",
            "seats",
            "service",
            "sia",
        }
    )
    text = " ".join(text_series.dropna().astype(str).tolist()).lower().strip()
    if not text:
        return None, {}

    tokens = re.findall(r"[A-Za-z']+", text)
    tokens = [token for token in tokens if token not in custom_stopwords]
    if not tokens:
        return None, {}

    frequencies = Counter(tokens)
    cloud = WordCloud(
        width=900,
        height=500,
        background_color="white",
        stopwords=custom_stopwords,
        collocations=False,
    ).generate_from_frequencies(frequencies)
    return cloud, frequencies


def linkify_wordcloud_svg(svg: str, frequencies):
    def replacer(match):
        attrs = match.group(1)
        word = match.group(2)
        count = frequencies.get(word, 0)
        url = f"?keyword={quote(word)}#recent-reviews"
        if 'style="' in attrs:
            attrs = re.sub(r'style="', 'style="cursor:pointer; ', attrs, count=1)
        else:
            attrs = f'{attrs} style="cursor:pointer;"'
        tooltip = escape(f"{word}: {count}")
        safe_word = escape(word)
        return (
            f'<a href="{url}" target="_top">'
            f"<text{attrs}><title>{tooltip}</title>{safe_word}</text></a>"
        )

    svg_body = svg.replace('<?xml version="1.0" encoding="UTF-8"?>', "").strip()
    return re.sub(r"<text([^>]*)>([^<]+)</text>", replacer, svg_body)


def render_interactive_cloud(wordcloud, frequencies):
    if wordcloud is None:
        return
    svg = linkify_wordcloud_svg(wordcloud.to_svg(), frequencies)
    st.markdown(svg, unsafe_allow_html=True)


@st.cache_data
def load_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = (
        df.columns.str.replace("\ufeff", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )

    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
        df["published_date"] = df["published_date"].dt.tz_convert(None)

    for col in ["rating", "helpful_votes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


data_path = Path(__file__).parent / "data" / "singapore_airlines_reviews.csv"
if not data_path.exists():
    st.error(f"Data file not found: {data_path}")
    st.stop()

df = load_reviews(data_path)

st.title("SIA Review Pulse")
st.caption("Singapore Airlines review insights from the latest dataset")

st.sidebar.header("Filters")
filtered = df.copy()

if "published_date" in filtered.columns:
    min_date = filtered["published_date"].min()
    max_date = filtered["published_date"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        use_full_range = st.sidebar.checkbox("Use full date range", value=False)
        default_end = max_date.date()
        default_start = (max_date - pd.DateOffset(months=12)).date()
        if default_start < min_date.date():
            default_start = min_date.date()

        if use_full_range:
            default_start = min_date.date()

        start_date = st.sidebar.date_input(
            "Start date",
            value=default_start,
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        end_date = st.sidebar.date_input(
            "End date",
            value=default_end,
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if start_date and end_date:
            start_date, end_date = sorted((start_date, end_date))
            mask = filtered["published_date"].dt.date.between(start_date, end_date)
            filtered = filtered[mask]

if "rating" in filtered.columns and filtered["rating"].notna().any():
    min_rating = int(filtered["rating"].min())
    max_rating = int(filtered["rating"].max())
    rating_range = st.sidebar.slider(
        "Rating range",
        min_value=min_rating,
        max_value=max_rating,
        value=(min_rating, max_rating),
    )
    filtered = filtered[filtered["rating"].between(rating_range[0], rating_range[1])]

if "published_platform" in filtered.columns:
    platforms = sorted(filtered["published_platform"].dropna().unique().tolist())
    if platforms:
        selected_platforms = st.sidebar.multiselect(
            "Platform",
            platforms,
            default=platforms,
        )
        filtered = filtered[filtered["published_platform"].isin(selected_platforms)]

if "type" in filtered.columns:
    types = sorted(filtered["type"].dropna().unique().tolist())
    if types:
        selected_types = st.sidebar.multiselect(
            "Review type",
            types,
            default=types,
        )
        filtered = filtered[filtered["type"].isin(selected_types)]

search_term = st.sidebar.text_input("Search in title or text")
if search_term:
    combined = pd.Series("", index=filtered.index)
    if "title" in filtered.columns:
        combined = combined.str.cat(filtered["title"].fillna(""), sep=" ")
    if "text" in filtered.columns:
        combined = combined.str.cat(filtered["text"].fillna(""), sep=" ")
    filtered = filtered[combined.str.contains(search_term, case=False, na=False)]

if filtered.empty:
    st.warning("No reviews match the selected filters.")
    st.stop()

if "rating" in filtered.columns and filtered["rating"].notna().any():
    avg_rating = filtered["rating"].mean()
    positive_share = (filtered["rating"].between(4, 5)).mean() * 100
    negative_share = (filtered["rating"].between(1, 2)).mean() * 100
    summary = (
        f"Showing {len(filtered):,} reviews for the selected filters with an average rating "
        f"of {avg_rating:.2f}. Positive reviews (4-5) make up {positive_share:.1f}% and "
        f"negative reviews (1-2) make up {negative_share:.1f}%."
    )
else:
    summary = f"Showing {len(filtered):,} reviews for the selected filters."

st.markdown(summary)

st.subheader("Predictions")
st.caption("Upcoming insights will appear here as forecasting models are added.")

metric_cols = st.columns(4)
metric_cols[0].metric("Total reviews", f"{len(filtered):,}")

if "rating" in filtered.columns and filtered["rating"].notna().any():
    avg_rating = filtered["rating"].mean()
    median_rating = filtered["rating"].median()
    high_rating_share = (filtered["rating"] >= 4).mean() * 100
    metric_cols[1].metric("Average rating", f"{avg_rating:.2f}")
    metric_cols[2].metric("Median rating", f"{median_rating:.0f}")
    metric_cols[3].metric("4-5 star share", f"{high_rating_share:.1f}%")
else:
    metric_cols[1].metric("Average rating", "N/A")
    metric_cols[2].metric("Median rating", "N/A")
    metric_cols[3].metric("4-5 star share", "N/A")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Rating distribution")
    if "rating" in filtered.columns and filtered["rating"].notna().any():
        rating_counts = filtered["rating"].value_counts().sort_index()
        st.bar_chart(rating_counts)
        st.caption("Shows how many reviews fall into each star rating in the current filters.")
    else:
        st.info("Ratings are missing for the current selection.")

with chart_right:
    st.subheader("Review volume over time")
    if "published_date" in filtered.columns and filtered["published_date"].notna().any():
        trend = (
            filtered.dropna(subset=["published_date"])
            .assign(month=lambda x: x["published_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month")
            .size()
            .rename("reviews")
            .to_frame()
        )
        st.line_chart(trend)
        st.caption("Monthly count of reviews, indicating volume trends over time.")
    else:
        st.info("Published dates are missing for the current selection.")

breakdown_left, breakdown_right = st.columns(2)

with breakdown_left:
    st.subheader("Platform breakdown")
    if "published_platform" in filtered.columns:
        platform_counts = filtered["published_platform"].value_counts()
        st.bar_chart(platform_counts)
        st.caption("Number of reviews by platform for the current filters.")
    else:
        st.info("Platform data is unavailable.")

with breakdown_right:
    st.subheader("Review type breakdown")
    if "type" in filtered.columns:
        type_counts = filtered["type"].value_counts()
        st.bar_chart(type_counts)
        st.caption("Counts of reviews by type (e.g., verified, trip type) in the selection.")
    else:
        st.info("Review type data is unavailable.")

st.subheader("Keyword clouds")
cloud_left, cloud_right = st.columns(2)

if "rating" in filtered.columns:
    positive = filtered[filtered["rating"].between(4, 5)]
    negative = filtered[filtered["rating"].between(1, 2)]
else:
    positive = pd.DataFrame()
    negative = pd.DataFrame()

def cloud_text(df_slice: pd.DataFrame) -> pd.Series:
    text_parts = []
    if "title" in df_slice.columns:
        text_parts.append(df_slice["title"])
    if "text" in df_slice.columns:
        text_parts.append(df_slice["text"])
    if not text_parts:
        return pd.Series([], dtype=str)
    combined = text_parts[0].fillna("").astype(str)
    for extra in text_parts[1:]:
        combined = combined.str.cat(extra.fillna("").astype(str), sep=" ")
    return combined

with cloud_left:
    st.subheader("Positive reviews (ratings 4-5)")
    if positive.empty:
        st.info("No positive reviews available for the current filters.")
    else:
        cloud, frequencies = build_keyword_cloud(cloud_text(positive))
        if cloud is None:
            st.info("Not enough text to build a word cloud.")
        else:
            render_interactive_cloud(cloud, frequencies)
            st.caption("Most frequent terms in positive review text after removing common words. Hover for counts; click to jump to reviews.")

with cloud_right:
    st.subheader("Negative reviews (ratings 1-2)")
    if negative.empty:
        st.info("No negative reviews available for the current filters.")
    else:
        cloud, frequencies = build_keyword_cloud(cloud_text(negative))
        if cloud is None:
            st.info("Not enough text to build a word cloud.")
        else:
            render_interactive_cloud(cloud, frequencies)
            st.caption("Most frequent terms in negative review text after removing common words. Hover for counts; click to jump to reviews.")

st.markdown('<div id="recent-reviews"></div>', unsafe_allow_html=True)
st.subheader("Recent reviews")
keyword_param = st.query_params.get("keyword")
if isinstance(keyword_param, list):
    keyword_value = keyword_param[0] if keyword_param else ""
else:
    keyword_value = keyword_param or ""
if keyword_value:
    st.info(f"Showing reviews containing '{keyword_value}'.")
    if st.button("Clear keyword filter"):
        st.query_params.clear()
        st.rerun()
preview_cols = [
    col
    for col in ["published_date", "rating", "title", "published_platform", "helpful_votes"]
    if col in filtered.columns
]

if "published_date" in filtered.columns:
    preview = filtered.sort_values("published_date", ascending=False)
else:
    preview = filtered

if keyword_value:
    combined = pd.Series("", index=preview.index)
    if "title" in preview.columns:
        combined = combined.str.cat(preview["title"].fillna(""), sep=" ")
    if "text" in preview.columns:
        combined = combined.str.cat(preview["text"].fillna(""), sep=" ")
    preview = preview[combined.str.contains(keyword_value, case=False, na=False)]
    if preview.empty:
        st.warning("No reviews match that keyword for the current filters.")
        st.stop()

st.dataframe(preview[preview_cols].head(20), width="stretch")
