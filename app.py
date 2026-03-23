import os
import json
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="NL to SQL + Spider Eval", layout="wide")
st.title("Natural Language to SQL + Spider Evaluation")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def build_sqlalchemy_uri(
    db_type: str,
    host: str,
    port: str,
    db_name: str,
    username: str,
    password: str,
) -> str:
    if db_type == "PostgreSQL":
        driver = "postgresql+psycopg2"
        default_port = "5432"
    else:
        driver = "mysql+mysqlconnector"
        default_port = "3306"

    host = (host or "").strip() or "localhost"
    port = (port or "").strip() or default_port
    db_name = (db_name or "").strip()
    username = (username or "").strip()
    password = password or ""

    return f"{driver}://{username}:{quote_plus(password)}@{host}:{port}/{db_name}"


@st.cache_resource(show_spinner=False)
def get_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True)


def render_df_download(df: pd.DataFrame, filename: str = "query_results.csv"):
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    app_mode = st.radio(
        "Mode",
        [
            "Live DB (PostgreSQL / MySQL)",
            "Spider Eval (SQLite)",
            "Evals Dashboard (CSV Compare)",
        ],
        index=0,
    )

    provider = st.selectbox(
        "Model backend",
        [
            "Gemini 2.5 Flash",
            "OpenAI GPT-4o-mini",
            "Mistral Large (API)",
            "Claude 3.5",
            "Ollama Llama3",
            "Local T5 (Spider)",
        ],
        index=0,
    )

    default_limit = st.number_input(
        "Default LIMIT (applied if not specified)",
        min_value=10,
        max_value=10000,
        value=200,
        step=10,
    )

    if app_mode.startswith("Live DB"):
        db_type = st.selectbox("Database type", ["PostgreSQL", "MySQL"], index=0)
        host = st.text_input("Host", value="localhost")
        port = st.text_input("Port", value="5432" if db_type == "PostgreSQL" else "3306")
        db_name = st.text_input("Database name", value="testDb" if db_type == "PostgreSQL" else "hr")
        username = st.text_input("Username", value="postgres" if db_type == "PostgreSQL" else "root")
        password = st.text_input("Password", type="password")

        max_semantic_retries = st.slider("Max semantic retries", 1, 5, 3)
        max_db_retries = st.slider("Max DB retries", 1, 5, 3)

        st.markdown("---")
        st.subheader("Phase 2")

        use_phase2 = st.checkbox("Enable Phase 2 workflow", value=False)
        kb_dir = st.text_input("KB directory", value="./kb")
        namespace = st.text_input("KB namespace", value="default")
        top_k = st.slider("Schema retrieval top-k", 3, 20, 8)
        max_iters = st.slider("Phase 2 max iterations", 2, 8, 4)

        learning_mode = st.selectbox("Learning mode", ["off", "propose", "always"], index=1)
        enable_result_validation = st.checkbox("Enable result validation", value=True)
        enable_join_validation = st.checkbox("Enable join-path validation", value=True)
        hitl_require_approval = st.checkbox("HITL: require approval before execution", value=False)

        st.markdown("---")
        st.subheader("Input Settings")

        input_source_mode = st.radio(
            "Input source",
            ["Auto (prefer text)", "Text only", "Voice only"],
            index=0,
        )

        enable_voice_input = st.checkbox("Enable voice input widget", value=True)
        input_language = st.selectbox(
            "Typed input language",
            ["auto", "english", "kannada", "hindi", "tamil", "telugu", "malayalam"],
            index=0,
        )

    elif app_mode.startswith("Spider Eval"):
        spider_root = st.text_input(
            "Spider root folder",
            value=os.getenv("SPIDER_ROOT", ""),
            placeholder=r"e.g. C:\datasets\spider",
        )
        split = st.selectbox("Split", ["dev", "train"], index=0)
        max_examples = st.number_input("Max examples", min_value=1, max_value=10000, value=100, step=10)
        db_id_filter = st.text_input("Optional db_id filter", value="")


# -----------------------------------------------------------------------------
# Live DB Mode
# -----------------------------------------------------------------------------
if app_mode.startswith("Live DB"):
    from phase2_types import Phase2Config
    from nl2sql_runner import (
        run_nl2sql_flow,
        run_nl2sql_flow_phase2,
        execute_sql_with_retries,
    )

    st.subheader("1) Provide schema / metadata")

    schema_mode = st.radio(
        "Schema input method",
        ["Paste schema text", "Upload .sql or .txt file"],
        index=0,
        horizontal=True,
    )

    schema_text = ""

    if schema_mode == "Paste schema text":
        schema_text = st.text_area(
            "Schema (tables, columns, relationships)",
            height=220,
            placeholder=(
                "Paste table definitions, columns, foreign keys, or DDL here.\n\n"
                "Example:\n"
                "Table customers(id, name, city)\n"
                "Table orders(id, customer_id, amount, dt)\n"
                "Relationship: orders.customer_id -> customers.id"
            ),
        )
    else:
        uploaded_file = st.file_uploader("Upload schema file", type=["sql", "txt"])
        if uploaded_file is not None:
            schema_text = uploaded_file.read().decode("utf-8", errors="replace")
            st.success(f"Loaded schema from file: {uploaded_file.name}")
            with st.expander("Preview uploaded schema"):
                st.code(schema_text[:4000], language="sql")

    st.subheader("2) Ask your question")

    typed_question = st.text_area(
        "Type your question",
        height=120,
        placeholder="Example: Show total revenue by company in 2024.",
    )

    voice_original_text = ""
    voice_detected_language = ""
    voice_english_text = ""

    if enable_voice_input:
        audio_question = st.audio_input("Or record your question")
        if audio_question is not None:
            try:
                from input_utils import transcribe_and_translate_audio_openai

                audio_bytes = audio_question.read()
                voice_payload = transcribe_and_translate_audio_openai(audio_bytes)

                voice_original_text = voice_payload.get("original_text", "").strip()
                voice_detected_language = voice_payload.get("detected_language", "unknown").strip()
                voice_english_text = voice_payload.get("english_text", "").strip()

                if voice_original_text:
                    st.info(f"Voice transcript ({voice_detected_language}): {voice_original_text}")

                if voice_english_text:
                    st.caption(f"Voice translated to English: {voice_english_text}")

            except Exception as e:
                st.warning(f"Voice transcription failed: {e}")

    # -------------------------------------------------------------------------
    # Resolve final NL question
    # -------------------------------------------------------------------------
    nl_question = ""
    source_used = ""

    typed_original = (typed_question or "").strip()

    if input_source_mode == "Text only":
        if typed_original:
            try:
                from input_utils import normalize_question_to_english

                typed_english, typed_lang = normalize_question_to_english(
                    provider_name=provider,
                    question=typed_original,
                    source_language=input_language,
                )

                st.info(f"Typed input ({typed_lang}): {typed_original}")
                if typed_english:
                    st.caption(f"Typed translated to English: {typed_english}")

                nl_question = typed_english or typed_original
                source_used = "text"
            except Exception as e:
                st.warning(f"Typed language normalization skipped: {e}")
                nl_question = typed_original
                source_used = "text"

    elif input_source_mode == "Voice only":
        nl_question = voice_english_text or ""
        source_used = "voice"

    else:
        # Auto: prefer typed text if present, else voice
        if typed_original:
            try:
                from input_utils import normalize_question_to_english

                typed_english, typed_lang = normalize_question_to_english(
                    provider_name=provider,
                    question=typed_original,
                    source_language=input_language,
                )

                st.info(f"Typed input ({typed_lang}): {typed_original}")
                if typed_english:
                    st.caption(f"Typed translated to English: {typed_english}")

                nl_question = typed_english or typed_original
                source_used = "text"
            except Exception as e:
                st.warning(f"Typed language normalization skipped: {e}")
                nl_question = typed_original
                source_used = "text"
        elif voice_english_text:
            nl_question = voice_english_text
            source_used = "voice"

    run_btn = st.button("Generate & Run SQL", type="primary")

    if run_btn:
        if not schema_text.strip():
            st.error("Please provide schema text.")
            st.stop()

        if not nl_question.strip():
            st.error("Please provide a text question or a voice input.")
            st.stop()

        st.success(f"Using {source_used} input for SQL generation.")
        st.code(nl_question, language="text")

        db_uri = build_sqlalchemy_uri(
            db_type=db_type,
            host=host,
            port=port,
            db_name=db_name,
            username=username,
            password=password,
        )

        try:
            engine = get_engine(db_uri)
        except Exception as e:
            st.error(f"Failed to create DB engine: {e}")
            st.stop()

        try:
            with st.spinner("Generating SQL and running query..."):
                if use_phase2:
                    cfg = Phase2Config(
                        kb_dir=kb_dir,
                        namespace=namespace,
                        top_k=int(top_k),
                        max_iters=int(max_iters),
                        default_limit=int(default_limit),
                        max_db_retries=int(max_db_retries),
                        enable_learning=(learning_mode != "off"),
                        learning_mode=str(learning_mode),
                        enable_result_validation=enable_result_validation,
                        enable_join_validation=enable_join_validation,
                        hitl_require_approval=hitl_require_approval,
                    )

                    result = run_nl2sql_flow_phase2(
                        engine=engine,
                        provider_name=provider,
                        db_type=db_type,
                        full_schema_text=schema_text,
                        nl_question=nl_question,
                        cfg=cfg,
                    )
                else:
                    result = run_nl2sql_flow(
                        engine=engine,
                        provider_name=provider,
                        db_type=db_type,
                        schema_text=schema_text,
                        nl_question=nl_question,
                        default_limit=int(default_limit),
                        max_semantic_retries=int(max_semantic_retries),
                        max_db_retries=int(max_db_retries),
                    )
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

        if result.get("status") == "awaiting_approval":
            st.warning("Phase 2 generated SQL and is waiting for manual approval before execution.")

            approved_sql = st.text_area(
                "Review / edit SQL before execution",
                value=result.get("sql", ""),
                height=180,
            )

            approved_params_text = st.text_area(
                "Review / edit params JSON",
                value=json.dumps(result.get("params", {}), ensure_ascii=False, indent=2),
                height=120,
            )

            if st.button("Execute approved SQL", type="primary"):
                try:
                    approved_params = json.loads(approved_params_text or "{}")
                    df_exec, db_tries = execute_sql_with_retries(
                        engine=engine,
                        sql_text=approved_sql,
                        params=approved_params,
                        max_db_retries=int(max_db_retries),
                    )
                    result["sql"] = approved_sql
                    result["params"] = approved_params
                    result["df"] = df_exec
                    result["db_retries_total"] = db_tries
                except Exception as e:
                    st.error(f"Approved SQL execution failed: {e}")
                    st.stop()
            else:
                st.stop()

        final_sql = result.get("sql", "")
        final_params = result.get("params", {})
        df = result.get("df", pd.DataFrame())
        logs = result.get("logs", [])
        semantic_attempts = result.get("semantic_attempts", 1)
        db_retries_total = result.get("db_retries_total", 1)
        dialect = result.get("dialect", db_type.lower())

        st.subheader("Query Results")

        with st.expander("Final SQL used", expanded=True):
            st.code(final_sql or "No valid SQL was produced.", language="sql")
            st.json(final_params or {})

        if result.get("phase") == "phase2":
            with st.expander("Phase 2: Retrieved schema pack"):
                st.code(result.get("schema_pack", ""), language="text")

            with st.expander("Phase 2: Decomposition plan"):
                st.json(result.get("plan", {}))

            with st.expander("Phase 2: Trace file path"):
                st.write(result.get("trace_path", ""))

            with st.expander("Phase 2: Result validation"):
                st.json(result.get("result_validation", {}))

            proposal = result.get("learning_proposal")
            if proposal:
                from schema_kb import append_learned_example

                st.info("Learning mode is PROPOSE. Review and save this successful example if valid.")
                if st.button("Save learned example"):
                    append_learned_example(
                        nl_question=proposal["nl_question"],
                        sql_text=proposal["sql_text"],
                        kb_dir=proposal["kb_dir"],
                        namespace=proposal["namespace"],
                    )
                    st.success("Learned example saved.")

        if df is None or df.empty:
            st.info(f"No rows returned after {semantic_attempts} semantic attempt(s).")
        else:
            st.dataframe(df, use_container_width=True, height=420)
            render_df_download(df, filename="query_results.csv")

        st.caption(
            f"Dialect: {dialect}, semantic attempts: {semantic_attempts}, total DB retries: {db_retries_total}."
        )

        with st.expander("Detailed execution log"):
            st.json(logs)

# -----------------------------------------------------------------------------
# Spider Eval Mode
# -----------------------------------------------------------------------------
elif app_mode.startswith("Spider Eval"):
    from spider_eval import EvalConfig, run_spider_eval

    st.subheader("Spider evaluation (execution accuracy)")
    run_eval_btn = st.button("Run Spider Evaluation", type="primary")

    if run_eval_btn:
        if not spider_root.strip():
            st.error("Please provide the Spider root folder.")
            st.stop()

        cfg = EvalConfig(
            spider_root=spider_root.strip(),
            split=split,
            provider_name=provider,
            default_limit=int(default_limit),
            max_examples=int(max_examples),
            db_id_filter=(db_id_filter.strip() or None),
        )

        try:
            with st.spinner("Running evaluation..."):
                df_res, summary = run_spider_eval(cfg)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()

        # st.success(
        #     f"Done. Accuracy = {summary['accuracy']:.3f} "
        #     f"({summary['correct']}/{summary['examples']}) on split={summary['split']}."
        # )

        # c1, c2 = st.columns(2)
        # with c1:
        #     st.metric("Execution Accuracy", f"{summary['accuracy'] * 100:.2f}%")
        # with c2:
        #     st.metric("Correct / Total", f"{summary['correct']} / {summary['examples']}")

        exec_acc = summary.get("execution_accuracy", summary.get("accuracy", 0.0))
        correct = summary.get("correct", 0)
        examples = summary.get("examples", 0)
        split_name = summary.get("split", split)

        st.success(
            f"Done. Execution Accuracy = {exec_acc:.3f} "
            f"({correct}/{examples}) on split={split_name}."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Execution Accuracy", f"{exec_acc * 100:.2f}%")
        with c2:
            st.metric("Correct / Total", f"{correct} / {examples}")

        st.dataframe(df_res, use_container_width=True, height=520)

        st.download_button(
            "Download evaluation CSV",
            data=df_res.to_csv(index=False).encode("utf-8"),
            file_name=f"spider_eval_{summary['split']}.csv",
            mime="text/csv",
        )

        with st.expander("Summary JSON"):
            st.json(summary)

# -----------------------------------------------------------------------------
# Eval Dashboard Mode
# -----------------------------------------------------------------------------
else:
    from phase2_eval_dashboard import render_eval_dashboard

    st.header("Spider Benchmark Evaluation Dashboard")
    render_eval_dashboard()