"""
Streamlit interface for the Fraud Detection MAS.

Run with:
    streamlit run app.py
"""

import io
import time

import pandas as pd
import plotly.express as px
import streamlit as st

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection MAS",
    page_icon="🔍",
    layout="wide",
)

# ── helpers ─────────────────────────────────────────────────────────────────

def _imports_ok() -> bool:
    try:
        import fraud_mas  # noqa: F401
        return True
    except ImportError as e:
        st.error(f"Cannot import fraud_mas package: {e}")
        return False


@st.cache_resource(show_spinner=False)
def _load_model_and_encoders():
    from fraud_mas.data_io import load_label_encoders, load_model
    model    = load_model()
    encoders = load_label_encoders()
    return model, encoders


def _model_trained() -> bool:
    from fraud_mas.config import XGB_MODEL_PATH, LABEL_ENC_PATH
    return XGB_MODEL_PATH.exists() and LABEL_ENC_PATH.exists()


# ── sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Fraud Detection MAS")
st.sidebar.markdown("Multi-Agent System with LLM Orchestration")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Train", "Detect", "SPADE Agents", "Results", "Memory"],
    index=0,
)

st.sidebar.divider()
if _model_trained():
    st.sidebar.success("Model: ready")
else:
    st.sidebar.warning("Model: not trained yet")

# ── Train page ───────────────────────────────────────────────────────────────
if page == "Train":
    st.title("Train Model")
    st.markdown("Upload a labelled CSV to train the XGBoost ensemble.")

    uploaded = st.file_uploader("Training CSV (must have a `label` column)", type="csv")

    label_col = st.text_input("Label column name", value="label")
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("XGBoost estimators", 100, 1000, 400, step=50)
    with col2:
        max_depth = st.slider("Max depth", 3, 10, 6)

    if st.button("Train", type="primary", disabled=uploaded is None):
        if not _imports_ok():
            st.stop()

        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df):,}** rows × **{df.shape[1]}** columns")

        if label_col not in df.columns:
            st.error(f"Column `{label_col}` not found. Available: {list(df.columns)}")
            st.stop()

        from fraud_mas.config import XGB_PARAMS
        XGB_PARAMS.update({"n_estimators": n_estimators, "max_depth": max_depth})

        from fraud_mas.behavioral import compute_behavioral_signals
        from fraud_mas.features import engineer_features
        from fraud_mas.geo import compute_geo_signals
        from fraud_mas.model import evaluate, predict_proba, train
        from fraud_mas.network_analysis import compute_network_signals
        from fraud_mas.nlp_risk import compute_nlp_signals
        from fraud_mas.data_io import save_label_encoders

        progress = st.progress(0, text="Running Agent 1 – Feature Engineering...")
        df, encoders = engineer_features(df, fit=True)
        save_label_encoders(encoders)
        progress.progress(20, text="Running Agent 2 – Behavioral...")
        df = compute_behavioral_signals(df)
        progress.progress(40, text="Running Agent 3 – Geo...")
        df = compute_geo_signals(df)
        progress.progress(60, text="Running Agent 4 – NLP...")
        df = compute_nlp_signals(df)
        progress.progress(80, text="Running Agent 5 – Network...")
        df = compute_network_signals(df)
        progress.progress(90, text="Training XGBoost ensemble...")

        model = train(df, label_col=label_col)
        scores = predict_proba(df, model)
        metrics = evaluate(df, scores, label_col=label_col)

        progress.progress(100, text="Done!")
        _load_model_and_encoders.clear()

        st.success("Model trained successfully!")
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC", metrics["roc_auc"])
        m2.metric("F1 Score", metrics["f1"])

        # Score distribution
        fig = px.histogram(
            x=scores, nbins=50,
            labels={"x": "Fraud probability"},
            title="Score distribution (training set)",
            color_discrete_sequence=["#e05c5c"],
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Detect page ──────────────────────────────────────────────────────────────
elif page == "Detect":
    st.title("Run Fraud Detection")
    st.markdown("Upload a test CSV and run all agents + LLM orchestration.")

    if not _model_trained():
        st.warning("No trained model found. Go to **Train** first.")
        st.stop()

    uploaded = st.file_uploader("Test CSV", type="csv")

    col1, col2 = st.columns(2)
    with col1:
        use_llm = st.toggle("Enable LLM review for borderline cases", value=True)
    with col2:
        id_col = st.text_input("Transaction ID column", value="transaction_id")

    if st.button("Run Detection", type="primary", disabled=uploaded is None):
        if not _imports_ok():
            st.stop()

        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df):,}** rows")

        model, encoders = _load_model_and_encoders()

        if not use_llm:
            # Monkey-patch llm_decide to skip API calls
            import fraud_mas.pipeline as _pipe
            _original_llm = _pipe.llm_decide

            def _no_llm(rows, **kwargs):
                return [{**r, "label": "fraud", "llm_reason": "LLM disabled"} for r in rows]

            _pipe.llm_decide = _no_llm

        with st.spinner("Running pipeline (agents 2-5 in parallel)..."):
            t0 = time.perf_counter()
            from fraud_mas.pipeline import run_pipeline, write_submission_file
            results = run_pipeline(df, model=model, encoders=encoders, verbose=False)
            elapsed = time.perf_counter() - t0

        if not use_llm:
            _pipe.llm_decide = _original_llm

        write_submission_file(results, id_col=id_col)
        st.session_state["results"] = results
        st.session_state["elapsed"] = elapsed

        st.success(f"Detection complete in {elapsed:.2f}s")

        fraud_n  = (results["label"] == "fraud").sum()
        legit_n  = (results["label"] == "legit").sum()
        review_n = (results["initial_label"] == "review").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total",    len(results))
        c2.metric("Fraud",    fraud_n,  delta=f"{fraud_n/len(results)*100:.1f}%", delta_color="inverse")
        c3.metric("Legit",    legit_n)
        c4.metric("LLM reviewed", review_n)

        st.dataframe(
            results[["transaction_id", "model_score", "initial_label", "label", "llm_reason"]]
            .sort_values("model_score", ascending=False)
            .head(200),
            use_container_width=True,
        )

        # Download button
        csv_bytes = results.to_csv(index=False).encode()
        st.download_button(
            "Download full results CSV",
            data=csv_bytes,
            file_name="fraud_results.csv",
            mime="text/csv",
        )


# ── SPADE Agents page ────────────────────────────────────────────────────────
elif page == "SPADE Agents":
    st.title("SPADE Multi-Agent System")
    st.markdown("Run the full agent pipeline over XMPP using the SPADE framework.")

    st.info(
        "**Prerequisites:** the XMPP server must be running.\n\n"
        "```bash\n"
        "docker compose up -d          # start prosody XMPP server\n"
        "python scripts/run_spade.py   # start all 8 SPADE agents\n"
        "```"
    )

    st.divider()
    st.subheader("Agent roster")

    from fraud_mas.config import AGENT_JIDS
    agent_rows = [{"Agent": k, "JID": v, "Role": {
        "orchestrator": "FSM coordinator – dispatches batches, collects results",
        "feature":      "Agent 1 – statistical feature engineering",
        "behavioral":   "Agent 2 – velocity, new-merchant, large-jump signals",
        "geo":          "Agent 3 – country risk, impossible travel",
        "nlp":          "Agent 4 – keyword risk, obfuscation scoring",
        "network":      "Agent 5 – shared device/IP, graph degree",
        "model":        "XGBoost ensemble scoring + threshold routing",
        "llm":          "Claude LLM – borderline case final decision",
    }.get(k, "")} for k, v in AGENT_JIDS.items()]
    st.dataframe(pd.DataFrame(agent_rows), use_container_width=True, hide_index=True)

    st.subheader("FSM flow")
    st.code(
        "IDLE → FEATURE → PARALLEL (behavioral‖geo‖nlp‖network) → SCORE → [LLM_REVIEW →] FINALIZE → IDLE",
        language=None,
    )

    st.divider()
    st.subheader("Run SPADE pipeline")

    if not _model_trained():
        st.warning("No trained model found. Go to **Train** first.")
        st.stop()

    uploaded_spade = st.file_uploader("Test CSV", type="csv", key="spade_upload")
    id_col_spade   = st.text_input("Transaction ID column", value="transaction_id", key="spade_id")

    if st.button("Run via SPADE agents", type="primary", disabled=uploaded_spade is None):
        if not _imports_ok():
            st.stop()

        df = pd.read_csv(uploaded_spade)
        st.write(f"Loaded **{len(df):,}** rows")

        try:
            import asyncio
            from fraud_mas.agents.spade_pipeline import run_spade_pipeline
            from fraud_mas.pipeline import write_submission_file

            with st.spinner("Agents communicating over XMPP..."):
                t0 = time.perf_counter()
                results = asyncio.run(run_spade_pipeline(df, verbose=False))
                elapsed = time.perf_counter() - t0

            write_submission_file(results, id_col=id_col_spade)
            st.session_state["results"] = results

            st.success(f"SPADE pipeline complete in {elapsed:.2f}s")

            fraud_n  = (results["label"] == "fraud").sum()
            legit_n  = (results["label"] == "legit").sum()
            review_n = (results.get("initial_label", pd.Series()) == "review").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total",        len(results))
            c2.metric("Fraud",        fraud_n,  delta=f"{fraud_n/len(results)*100:.1f}%", delta_color="inverse")
            c3.metric("Legit",        legit_n)
            c4.metric("LLM reviewed", review_n)

            st.dataframe(
                results[["transaction_id", "model_score", "initial_label", "label", "llm_reason"]]
                .sort_values("model_score", ascending=False)
                .head(200),
                use_container_width=True,
            )

            st.download_button(
                "Download results CSV",
                data=results.to_csv(index=False).encode(),
                file_name="spade_results.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"SPADE pipeline error: {exc}")
            st.markdown(
                "Make sure the XMPP server and agents are running:\n"
                "```bash\n"
                "docker compose up -d\n"
                "python scripts/run_spade.py\n"
                "```"
            )


# ── Results page ─────────────────────────────────────────────────────────────
elif page == "Results":
    st.title("Results Explorer")

    results = st.session_state.get("results")
    if results is None:
        # Try loading submission file
        from fraud_mas.config import SUBMISSION_PATH
        if SUBMISSION_PATH.exists():
            results = pd.read_csv(SUBMISSION_PATH)
            st.info("Loaded last submission file.")
        else:
            st.info("No results yet. Run detection first.")
            st.stop()

    tab1, tab2 = st.tabs(["Score distribution", "Agent signals"])

    with tab1:
        if "model_score" in results.columns:
            fig = px.histogram(
                results, x="model_score", color="label",
                nbins=60,
                barmode="overlay",
                color_discrete_map={"fraud": "#e05c5c", "legit": "#5c8fe0", "review": "#f0a500"},
                title="Model score distribution by final label",
            )
            st.plotly_chart(fig, use_container_width=True)

        counts = results["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        fig2 = px.pie(counts, names="label", values="count",
                      color="label",
                      color_discrete_map={"fraud": "#e05c5c", "legit": "#5c8fe0"},
                      title="Fraud vs Legit breakdown")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        agent_cols = {
            "Behavioral": "behav_score",
            "Geo":        "geo_score",
            "NLP":        "nlp_score",
            "Network":    "net_score",
        }
        available = {k: v for k, v in agent_cols.items() if v in results.columns}
        if not available:
            st.info("Agent signal columns not found in results.")
        else:
            for agent_name, col in available.items():
                fig = px.box(results, x="label", y=col, color="label",
                             color_discrete_map={"fraud": "#e05c5c", "legit": "#5c8fe0"},
                             title=f"{agent_name} score by label")
                st.plotly_chart(fig, use_container_width=True)


# ── Memory page ───────────────────────────────────────────────────────────────
elif page == "Memory":
    st.title("Fraud Memory")
    st.markdown("Patterns learned by the LLM orchestrator over time.")

    from fraud_mas.data_io import load_fraud_memory, save_fraud_memory

    memory = load_fraud_memory()

    patterns = memory.get("fraud_patterns", [])
    st.metric("Stored patterns", len(patterns))

    if patterns:
        st.dataframe(
            pd.DataFrame({"pattern": patterns}),
            use_container_width=True,
        )
    else:
        st.info("No fraud patterns in memory yet. Run detection with LLM enabled.")

    if st.button("Clear memory", type="secondary"):
        save_fraud_memory({})
        st.success("Memory cleared.")
        st.rerun()
