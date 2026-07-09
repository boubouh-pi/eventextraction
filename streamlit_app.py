"""
Streamlit demo for the event-extraction pipeline.

Two backends:
- API mode (API_URL env set): POSTs to the deployed FastAPI service —
  demonstrates the real production path.
- Local mode (default): runs the pipeline in-process, no server needed.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Event Extraction Demo", page_icon="📅", layout="wide")

API_URL = os.getenv("API_URL", "").rstrip("/")

METHOD_BADGES = {
    "ml": "🟢 ML",
    "llm_fallback": "🟣 LLM fallback",
    "hybrid": "🟡 Hybrid",
}


@st.cache_resource
def get_local_pipeline():
    from app.pipeline import EventExtractionPipeline
    return EventExtractionPipeline()


def extract_via_api(html: str, site: str, use_llm: bool) -> dict:
    r = requests.post(
        f"{API_URL}/extract",
        json={"html": html, "site": site or None, "use_llm_fallback": use_llm},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def extract_locally(html: str, use_llm: bool) -> dict:
    pipeline = get_local_pipeline()
    start = time.perf_counter()
    result = pipeline.extract(html, use_llm_fallback=use_llm)
    latency = (time.perf_counter() - start) * 1000
    return {
        "events": result.events,
        "n_nodes": result.n_nodes,
        "n_candidates": result.n_candidates,
        "extraction_method": result.extraction_method,
        "llm_cost_usd": result.llm_cost_usd,
        "latency_ms": round(latency, 1),
    }


st.title("📅 Event Extraction — ML + LLM Fallback")
st.caption(
    "LightGBM node detection → DOM clustering → confidence-gated LLM fallback. "
    f"Backend: {'API @ ' + API_URL if API_URL else 'local pipeline'}"
)

with st.sidebar:
    st.header("Options")
    use_llm = st.toggle("LLM fallback", value=True,
                        help="Escalate low-confidence clusters to GPT-4o(-mini)")
    site_hint = st.text_input("Site label (monitoring)", value="demo")

tab_url, tab_html = st.tabs(["From URL", "Paste HTML"])

html_input = None
with tab_url:
    url = st.text_input("Page URL", placeholder="https://example.org/college-fairs")
    if st.button("Fetch & extract", type="primary", disabled=not url):
        try:
            resp = requests.get(url, timeout=30,
                                headers={"User-Agent": "Mozilla/5.0 (event-extractor demo)"})
            resp.raise_for_status()
            html_input = resp.text
        except requests.RequestException as exc:
            st.error(f"Could not fetch URL: {exc}")

with tab_html:
    pasted = st.text_area("Raw HTML", height=240,
                          placeholder="<html>...</html>")
    if st.button("Extract from HTML", type="primary", disabled=not pasted):
        html_input = pasted

if html_input:
    with st.spinner("Extracting events…"):
        try:
            if API_URL:
                result = extract_via_api(html_input, site_hint, use_llm)
            else:
                result = extract_locally(html_input, use_llm)
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
            st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Events", len(result["events"]))
    c2.metric("DOM nodes", result["n_nodes"])
    c3.metric("Candidates", result["n_candidates"])
    c4.metric("Latency", f"{result['latency_ms']:.0f} ms")
    c5.metric("LLM cost", f"${result['llm_cost_usd']:.4f}")

    st.subheader(f"Method: {METHOD_BADGES.get(result['extraction_method'], '?')}")

    if not result["events"]:
        st.info("No events found on this page.")
    else:
        df = pd.DataFrame(result["events"])
        df["method"] = df["extraction_method"].map(METHOD_BADGES)
        st.dataframe(
            df[["name", "date", "time", "location", "confidence_score", "method"]],
            use_container_width=True, hide_index=True,
        )

        low_conf = df[df["confidence_score"] < 0.6]
        if not low_conf.empty and not use_llm:
            st.warning(
                f"{len(low_conf)} event(s) below confidence 0.6 — "
                "enable the LLM fallback to repair suspect clusters."
            )
