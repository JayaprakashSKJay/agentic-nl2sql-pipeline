from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from spider_metrics import analyze_eval_frame

ENRICHED_DIR = Path("spider_testing/enriched")
DASHBOARD_DIR = Path("spider_testing/dashboard")


# -----------------------------------------------------------------------------
# Visual theme / color maps
# -----------------------------------------------------------------------------
SUMMARY_METRIC_COLORS = {
    "execution_accuracy": "#2E8B57",
    "exact_match_accuracy": "#4169E1",
    "sql_validity_rate": "#20B2AA",
    "component_match_score": "#8A2BE2",
}

OUTCOME_COLORS = {
    "correct": "#2E8B57",
    "result_mismatch": "#FF8C00",
    "execution_error": "#DC143C",
    "gold_error": "#8B0000",
    "provider_error": "#6A5ACD",
    "parse_error": "#FF1493",
    "other": "#708090",
}

HARDNESS_COLORS = {
    "easy": "#2E8B57",
    "medium": "#FFD700",
    "hard": "#FF8C00",
    "extra_hard": "#DC143C",
}

COMPONENT_COLORS = {
    "SELECT": "#1f77b4",
    "WHERE": "#ff7f0e",
    "GROUP BY": "#2ca02c",
    "ORDER BY": "#d62728",
    "JOIN": "#9467bd",
}


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------
def _list_enriched_files() -> List[Path]:
    if not ENRICHED_DIR.exists():
        return []
    return sorted(ENRICHED_DIR.glob("*_enriched.csv"))


def _existing_chart_paths(run_name: str) -> Dict[str, Path]:
    base = DASHBOARD_DIR / f"{run_name}_charts"
    return {
        "base": base,
        "execution_accuracy": base / "execution_accuracy.html",
        "benchmark_metrics": base / "benchmark_metrics.html",
        "latency": base / "latency.html",
        "outcome_distribution": base / "outcome_distribution.html",
        "component_accuracy": base / "component_accuracy.html",
        "hardness_breakdown": base / "hardness_breakdown.html",
        "join_accuracy": base / "join_accuracy.html",
        "db_accuracy": base / "db_accuracy.html",
        "latency_histogram": base / "latency_histogram.html",
        "join_curve": base / "join_curve.html",
    }


def _all_charts_exist(run_name: str) -> bool:
    paths = _existing_chart_paths(run_name)
    return all(v.exists() for k, v in paths.items() if k != "base")


def _save_fig_html(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(path), auto_open=False, include_plotlyjs="cdn")


def _render_saved_html(path: Path, height: int = 500) -> None:
    html = path.read_text(encoding="utf-8")
    st.components.v1.html(html, height=height, scrolling=True)


def _apply_common_layout(fig, title: str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        title_x=0.02,
        font=dict(size=13),
        margin=dict(l=40, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# -----------------------------------------------------------------------------
# Chart builders
# -----------------------------------------------------------------------------
def _build_summary_figs(summary: dict):
    single_df = pd.DataFrame(
        [
            {
                "run_name": summary["run_name"],
                "execution_accuracy": summary["execution_accuracy"],
                "exact_match_accuracy": summary["exact_match_accuracy"],
                "sql_validity_rate": summary["sql_validity_rate"],
                "component_match_score": summary["component_match_score"],
                "avg_latency_ms": summary["avg_latency_ms"],
            }
        ]
    )

    fig_acc = px.bar(
        single_df,
        x="run_name",
        y="execution_accuracy",
        text="execution_accuracy",
        color_discrete_sequence=[SUMMARY_METRIC_COLORS["execution_accuracy"]],
    )
    fig_acc.update_yaxes(range=[0, 1], title="Accuracy")
    _apply_common_layout(fig_acc, "Execution Accuracy")

    compare_df = single_df.melt(
        id_vars=["run_name"],
        value_vars=[
            "execution_accuracy",
            "exact_match_accuracy",
            "sql_validity_rate",
            "component_match_score",
        ],
        var_name="metric",
        value_name="value",
    )

    compare_df["metric_label"] = compare_df["metric"].map(
        {
            "execution_accuracy": "Execution Accuracy",
            "exact_match_accuracy": "Exact Match",
            "sql_validity_rate": "SQL Validity",
            "component_match_score": "Component Match",
        }
    )

    fig_compare = px.bar(
        compare_df,
        x="metric_label",
        y="value",
        text="value",
        color="metric",
        color_discrete_map=SUMMARY_METRIC_COLORS,
    )
    fig_compare.update_yaxes(range=[0, 1], title="Score")
    fig_compare.update_xaxes(title="")
    _apply_common_layout(fig_compare, "Benchmark Metrics Summary")

    latency_df = pd.DataFrame(
        [
            {"metric": "Average", "value": summary["avg_latency_ms"]},
            {"metric": "Median", "value": summary["median_latency_ms"]},
            {"metric": "P95", "value": summary["p95_latency_ms"]},
        ]
    )

    fig_latency = px.bar(
        latency_df,
        x="metric",
        y="value",
        text="value",
        color="metric",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#9467bd"],
    )
    fig_latency.update_yaxes(title="Latency (ms)")
    fig_latency.update_xaxes(title="")
    _apply_common_layout(fig_latency, "Latency Summary")

    return fig_acc, fig_compare, fig_latency


def _build_run_figs(payload: dict):
    enriched_df = payload["enriched_df"]
    per_db_df = payload["per_db_df"]
    join_df = payload["join_df"]
    complexity_df = payload["complexity_df"]
    outcome_df = payload["outcome_df"]
    component_df = payload["component_df"]

    hardness_order = ["easy", "medium", "hard", "extra_hard"]
    if not complexity_df.empty:
        complexity_df = complexity_df.copy()
        complexity_df["complexity"] = pd.Categorical(
            complexity_df["complexity"],
            categories=hardness_order,
            ordered=True,
        )
        complexity_df = complexity_df.sort_values("complexity")

    fig_outcome = px.pie(
        outcome_df,
        names="outcome_type",
        values="count",
        color="outcome_type",
        color_discrete_map=OUTCOME_COLORS,
        hole=0.35,
    )
    _apply_common_layout(fig_outcome, "Outcome Distribution")

    fig_component = px.bar(
        component_df,
        x="component",
        y="accuracy",
        text="accuracy",
        color="component",
        color_discrete_map=COMPONENT_COLORS,
    )
    fig_component.update_yaxes(range=[0, 1], title="Accuracy")
    fig_component.update_xaxes(title="")
    _apply_common_layout(fig_component, "Component Matching Accuracy")

    fig_hardness = px.bar(
        complexity_df,
        x="complexity",
        y="execution_accuracy",
        text="execution_accuracy",
        color="complexity",
        color_discrete_map=HARDNESS_COLORS,
    )
    fig_hardness.update_yaxes(range=[0, 1], title="Accuracy")
    fig_hardness.update_xaxes(title="Hardness")
    _apply_common_layout(fig_hardness, "Spider Hardness Breakdown")

    fig_join_bar = px.bar(
        join_df,
        x="join_count",
        y="execution_accuracy",
        text="execution_accuracy",
        color="join_count",
        color_continuous_scale="Blues",
    )
    fig_join_bar.update_yaxes(range=[0, 1], title="Accuracy")
    fig_join_bar.update_xaxes(title="Join Count")
    _apply_common_layout(fig_join_bar, "Accuracy by Join Count")

    fig_join_curve = px.line(
        join_df,
        x="join_count",
        y="execution_accuracy",
        markers=True,
    )
    fig_join_curve.update_traces(line=dict(color="#ff7f0e", width=3), marker=dict(size=9, color="#d62728"))
    fig_join_curve.update_yaxes(range=[0, 1], title="Accuracy")
    fig_join_curve.update_xaxes(title="Join Count")
    _apply_common_layout(fig_join_curve, "Accuracy vs Join Count Curve")

    domain_df = per_db_df.rename(columns={"db_id": "domain"}).copy()
    fig_db = px.bar(
        domain_df,
        x="domain",
        y="execution_accuracy",
        hover_data=["examples", "correct", "avg_latency_ms"],
        color="execution_accuracy",
        color_continuous_scale="Viridis",
    )
    fig_db.update_yaxes(range=[0, 1], title="Accuracy")
    fig_db.update_xaxes(title="Domain / Database")
    _apply_common_layout(fig_db, "Per-Domain Performance")

    fig_hist = px.histogram(
        enriched_df,
        x="latency_ms",
        nbins=30,
        color_discrete_sequence=["#6A5ACD"],
    )
    fig_hist.update_xaxes(title="Latency (ms)")
    fig_hist.update_yaxes(title="Count")
    _apply_common_layout(fig_hist, "Latency Distribution")

    return {
        "outcome_distribution": fig_outcome,
        "component_accuracy": fig_component,
        "hardness_breakdown": fig_hardness,
        "join_accuracy": fig_join_bar,
        "join_curve": fig_join_curve,
        "db_accuracy": fig_db,
        "latency_histogram": fig_hist,
        "complexity_df": complexity_df,
        "domain_df": domain_df,
    }


def _save_run_charts(run_name: str, payload: dict, overwrite: bool = False) -> Path:
    paths = _existing_chart_paths(run_name)
    base = paths["base"]
    base.mkdir(parents=True, exist_ok=True)

    if _all_charts_exist(run_name) and not overwrite:
        return base

    summary = payload["summary"]
    fig_acc, fig_compare, fig_latency = _build_summary_figs(summary)
    _save_fig_html(fig_acc, paths["execution_accuracy"])
    _save_fig_html(fig_compare, paths["benchmark_metrics"])
    _save_fig_html(fig_latency, paths["latency"])

    run_figs = _build_run_figs(payload)
    _save_fig_html(run_figs["outcome_distribution"], paths["outcome_distribution"])
    _save_fig_html(run_figs["component_accuracy"], paths["component_accuracy"])
    _save_fig_html(run_figs["hardness_breakdown"], paths["hardness_breakdown"])
    _save_fig_html(run_figs["join_accuracy"], paths["join_accuracy"])
    _save_fig_html(run_figs["join_curve"], paths["join_curve"])
    _save_fig_html(run_figs["db_accuracy"], paths["db_accuracy"])
    _save_fig_html(run_figs["latency_histogram"], paths["latency_histogram"])

    return base


def _show_saved_or_generate(
    run_name: str,
    payload: dict,
    use_saved_if_exists: bool,
    save_generated: bool,
    overwrite_saved: bool,
) -> None:
    paths = _existing_chart_paths(run_name)

    if use_saved_if_exists and _all_charts_exist(run_name):
        st.caption(f"Showing saved charts from: {paths['base'].resolve()}")
        _render_saved_html(paths["execution_accuracy"], height=420)
        _render_saved_html(paths["benchmark_metrics"], height=500)
        _render_saved_html(paths["latency"], height=420)
        _render_saved_html(paths["outcome_distribution"], height=500)
        _render_saved_html(paths["component_accuracy"], height=500)
        _render_saved_html(paths["hardness_breakdown"], height=500)
        _render_saved_html(paths["join_accuracy"], height=500)
        _render_saved_html(paths["join_curve"], height=500)
        _render_saved_html(paths["db_accuracy"], height=600)
        _render_saved_html(paths["latency_histogram"], height=500)
        return

    st.caption("Saved charts not found for this run. Generating from enriched CSV.")

    summary = payload["summary"]
    fig_acc, fig_compare, fig_latency = _build_summary_figs(summary)

    st.plotly_chart(fig_acc, use_container_width=True, key=f"{run_name}_fig_acc")
    st.plotly_chart(fig_compare, use_container_width=True, key=f"{run_name}_fig_compare")
    st.plotly_chart(fig_latency, use_container_width=True, key=f"{run_name}_fig_latency")

    run_figs = _build_run_figs(payload)

    st.plotly_chart(run_figs["outcome_distribution"], use_container_width=True, key=f"{run_name}_fig_outcome")
    st.plotly_chart(run_figs["component_accuracy"], use_container_width=True, key=f"{run_name}_fig_component")
    st.plotly_chart(run_figs["hardness_breakdown"], use_container_width=True, key=f"{run_name}_fig_hardness")
    st.plotly_chart(run_figs["join_accuracy"], use_container_width=True, key=f"{run_name}_fig_join_bar")
    st.plotly_chart(run_figs["join_curve"], use_container_width=True, key=f"{run_name}_fig_join_curve")
    st.plotly_chart(run_figs["db_accuracy"], use_container_width=True, key=f"{run_name}_fig_db")
    st.plotly_chart(run_figs["latency_histogram"], use_container_width=True, key=f"{run_name}_fig_hist")

    if save_generated:
        saved_dir = _save_run_charts(run_name, payload, overwrite=overwrite_saved)
        st.success(f"Charts saved to: {saved_dir.resolve()}")


# -----------------------------------------------------------------------------
# Main dashboard renderer
# -----------------------------------------------------------------------------
def render_eval_dashboard() -> None:
    st.caption(f"Enriched folder: {ENRICHED_DIR.resolve() if ENRICHED_DIR.exists() else ENRICHED_DIR}")
    st.caption(f"Dashboard folder: {DASHBOARD_DIR.resolve() if DASHBOARD_DIR.exists() else DASHBOARD_DIR}")

    mode = st.radio(
        "Choose data source",
        ["Select file from spider_testing/enriched", "Upload a file"],
        horizontal=True,
    )

    selected_name = None
    selected_df = None

    if mode == "Select file from spider_testing/enriched":
        files = _list_enriched_files()
        if not files:
            st.warning("No enriched CSV files found in spider_testing/enriched.")
            return

        file_map = {p.name: p for p in files}
        selected_file_name = st.selectbox("Select enriched CSV file", list(file_map.keys()))
        selected_path = file_map[selected_file_name]

        selected_name = selected_path.stem.replace("_enriched", "")
        selected_df = pd.read_csv(selected_path)

    else:
        up = st.file_uploader("Upload enriched Spider CSV", type=["csv"])
        if up is None:
            st.info("Upload one enriched CSV file to continue.")
            return

        selected_name = up.name.replace(".csv", "").replace("_enriched", "")
        selected_df = pd.read_csv(up)

    payload = analyze_eval_frame(selected_df, run_name=selected_name)

    summary = payload["summary"]
    enriched_df = payload["enriched_df"]
    run_figs = _build_run_figs(payload)

    st.subheader(f"Detailed Summary — {selected_name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Execution Accuracy", f"{summary['execution_accuracy'] * 100:.2f}%")
    c2.metric("Exact Match", f"{summary['exact_match_accuracy'] * 100:.2f}%")
    c3.metric("SQL Validity", f"{summary['sql_validity_rate'] * 100:.2f}%")
    c4.metric("Component Match", f"{summary['component_match_score'] * 100:.2f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Avg Latency", f"{summary['avg_latency_ms']:.2f} ms")
    c6.metric("Median Latency", f"{summary['median_latency_ms']:.2f} ms")
    c7.metric("P95 Latency", f"{summary['p95_latency_ms']:.2f} ms")

    use_saved_if_exists = st.checkbox("Load saved charts if they already exist", value=True)
    save_generated = st.checkbox("Save generated charts to spider_testing/dashboard", value=True)
    overwrite_saved = st.checkbox("Overwrite existing saved charts", value=False)

    if st.button("Show / Generate Charts", type="primary"):
        _show_saved_or_generate(
            run_name=selected_name,
            payload=payload,
            use_saved_if_exists=use_saved_if_exists,
            save_generated=save_generated,
            overwrite_saved=overwrite_saved,
        )

    st.subheader("Spider Hardness Leaderboard Table")
    st.dataframe(
        run_figs["complexity_df"].rename(columns={"complexity": "hardness"}),
        use_container_width=True,
    )

    st.subheader("Per-Domain Performance Table")
    st.dataframe(
        run_figs["domain_df"],
        use_container_width=True,
    )

    with st.expander("Detailed enriched per-query results"):
        st.dataframe(enriched_df, use_container_width=True, height=500)

    st.download_button(
        "⬇️ Download enriched detailed CSV",
        data=enriched_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_name}_enriched.csv",
        mime="text/csv",
    )