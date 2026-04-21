import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import streamlit as st
from dotenv import load_dotenv

from src.retrieval.retrieval import retrieval
from src.generate.generate import generate
from src.utils.mongodb import mongo
from eval.evaluate import Evaluator

load_dotenv()


def run_async(coro):
    """Crea un event loop fresco y reconecta mongo para compatibilidad con Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    mongo._client = None
    mongo.connect()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

SOURCES = ["All", "govuk", "wto", "fsa"]
TOPICS  = [
    "All", "beverages", "customs", "tariff",
    "china_import", "robotics", "investment",
    "trade_policy", "china_trade_policy",
    "uk_trade_policy", "hong_kong_trade", "eu_trade_policy",
    "employment",
]
MODES = ["hybrid_rerank", "hybrid", "semantic"]

st.set_page_config(page_title="Clara — LEC Knowledge Assistant", layout="wide")

# ── Tabs ─────────────────────────────────────────────

tab_chat, tab_eval, tab_status = st.tabs(["Chat", "Evaluation", "Ingesta Status"])


# ════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════

with tab_chat:
    with st.sidebar:
        st.title("Filters")
        source = st.selectbox("Source", SOURCES)
        topic  = st.selectbox("Topic",  TOPICS)
        mode   = st.selectbox("Search mode", MODES)
        top_k  = st.slider("Results", min_value=1, max_value=10, value=5)
        st.divider()
        st.caption("LEC Trade Intelligence Platform")

    metadata_filter = {}
    if source != "All":
        metadata_filter["source"] = source
    if topic != "All":
        metadata_filter["topic"] = topic
    if not metadata_filter:
        metadata_filter = None

    st.title("Clara")
    st.caption("Your LEC knowledge assistant. Ask me anything about UK trade regulations, import duties, beverages licensing, and more.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm Clara, your LEC knowledge assistant."}
        ]

    messages_container = st.container(height=520)
    with messages_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    query = st.chat_input("Ask a question...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with messages_container:
            with st.chat_message("user"):
                st.markdown(query)

        with messages_container:
         with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                if mode == "semantic":
                    results = run_async(retrieval.semantic_search(query=query, top_k=top_k, metadata_filter=metadata_filter))
                elif mode == "hybrid":
                    results = run_async(retrieval.hybrid_search(query=query, top_k=top_k, metadata_filter=metadata_filter))
                else:
                    results = run_async(retrieval.hybrid_rerank_search(query=query, top_k=top_k, metadata_filter=metadata_filter))

            with st.spinner("Generating answer..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[1:-1]  # skip welcome msg and current query
                ]
                response = generate.answer(query=query, results=results, history=history)

            st.markdown(response.content)

            with st.expander(f"Sources ({len(results)}) — ${response.cost_usd:.6f} · {response.input_tokens + response.output_tokens} tokens"):
                for i, r in enumerate(results, 1):
                    title        = r.metadata.get("title", r.doc_id)
                    url          = r.metadata.get("url", "")
                    source_label = r.metadata.get("source", "").upper()
                    topic_label  = r.metadata.get("topic", "")
                    st.markdown(f"**[{i}] {title}** `{source_label}` `{topic_label}`")
                    if url:
                        st.caption(url)
                    st.markdown(f"> {r.text[:400]}{'...' if len(r.text) > 400 else ''}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("BM25",     f"{r.scores.bm25:.3f}")
                    c2.metric("Semantic", f"{r.scores.semantic:.3f}")
                    c3.metric("Reranker", f"{r.scores.reranker:.3f}")
                    st.divider()

            st.session_state.messages.append({"role": "assistant", "content": response.content})


# ════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ════════════════════════════════════════════════════

with tab_eval:
    st.title("Evaluation Metrics")
    st.caption("Runs precision@5, recall@5 and NDCG@5 across 3 search configurations.")

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Running 3 modes × 20 queries..."):
            evaluator = Evaluator()
            output = run_async(evaluator.run())

        st.success(f"Done — {output['n_queries']} queries evaluated at top-{output['top_k']}")

        rows = []
        for mode_name, metrics in output["results"].items():
            rows.append({"Mode": mode_name, **metrics})

        st.dataframe(rows, use_container_width=True)

        best_mode = max(output["results"], key=lambda m: output["results"][m]["ndcg@5"])
        st.info(f"Best mode by NDCG@5: **{best_mode}**")


# ════════════════════════════════════════════════════
# TAB 3 — INGESTA STATUS
# ════════════════════════════════════════════════════

with tab_status:
    st.title("Ingesta Status")

    if st.button("Refresh", type="secondary"):
        st.rerun()

    async def get_stats():
        col = mongo.get_collection()
        total_chunks = await col.count_documents({})

        by_src_chunks = {}
        by_src_docs   = {}
        for src in ["govuk", "wto", "fsa"]:
            by_src_chunks[src] = await col.count_documents({"metadata.source": src})
            by_src_docs[src]   = len(await col.distinct("doc_id", {"metadata.source": src}))

        total_docs = len(await col.distinct("doc_id"))

        topics = await col.distinct("metadata.topic")
        topic_counts = {}
        for t in topics:
            topic_counts[t] = await col.count_documents({"metadata.topic": t})

        return total_chunks, total_docs, by_src_chunks, by_src_docs, topic_counts

    total_chunks, total_docs, by_src_chunks, by_src_docs, topic_counts = run_async(get_stats())

    st.subheader("Documents in MongoDB")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total docs",  total_docs)
    c2.metric("GOV.UK docs", by_src_docs.get("govuk", 0))
    c3.metric("WTO docs",    by_src_docs.get("wto",   0))
    c4.metric("FSA docs",    by_src_docs.get("fsa",   0))

    st.subheader("Chunks in MongoDB")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total chunks",  total_chunks)
    c2.metric("GOV.UK chunks", by_src_chunks.get("govuk", 0))
    c3.metric("WTO chunks",    by_src_chunks.get("wto",   0))
    c4.metric("FSA chunks",    by_src_chunks.get("fsa",   0))

    st.subheader("Chunks by Topic")
    topic_rows = [{"Topic": t, "Chunks": n} for t, n in sorted(topic_counts.items(), key=lambda x: -x[1])]
    st.dataframe(topic_rows, use_container_width=True)

    st.subheader("Files on Disk vs MongoDB")
    corps = Path("corps")
    file_counts = {}
    for folder in ["govuk", "wto", "fsa"]:
        p = corps / folder
        file_counts[folder] = len(list(p.glob("*"))) if p.exists() else 0

    total_files = sum(file_counts.values())
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total files", total_files, delta=f"{total_docs - total_files:+d} vs MongoDB")
    f2.metric("GOV.UK files", file_counts["govuk"], delta=f"{by_src_docs.get('govuk',0) - file_counts['govuk']:+d}")
    f3.metric("WTO files",    file_counts["wto"],   delta=f"{by_src_docs.get('wto',0)   - file_counts['wto']:+d}")
    f4.metric("FSA files",    file_counts["fsa"],   delta=f"{by_src_docs.get('fsa',0)   - file_counts['fsa']:+d}")
