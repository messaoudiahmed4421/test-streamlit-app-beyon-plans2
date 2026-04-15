"""
Chat-like Streamlit interface for the P&L multi-agent pipeline.
Designed for easy integration with the notebook logic and Streamlit Cloud deployment.
Run with: streamlit run web_interface.py
"""

from datetime import datetime
from io import BytesIO
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline_runner_adk import run_pipeline_adk as execute_multi_agent_pipeline


APP_VERSION = "2.0-chat-workspace"

NAV_ITEMS = [
    "Executive Workspace",
    "Executive Dashboard",
    "Run History",
    "Performance Insights",
    "Operational Logs",
]

AGENTS_DATA = pd.DataFrame(
    {
        "Agent": ["A1", "A2", "A3", "A4", "A5"],
        "Statut": ["OK", "OK", "OK", "ERROR", "WAITING"],
        "Succes": [100, 92, 88, 50, 40],
        "Duree_s": [1.3, 2.7, 3.6, 4.1, 0.4],
    }
)

QUALITY_DATA = pd.DataFrame(
    {
        "Run": [1, 2, 3, 4, 5],
        "Date": ["23/03", "24/03", "25/03", "26/03", "27/03"],
        "Score Global": [5.5, 6.1, 6.4, 6.9, 7.2],
        "Actionnabilite": [3.0, 4.0, 5.0, 5.5, 6.0],
    }
)

STATIC_LOGS = [
    ("11:49:10", "A1", "SUCCESS", "DatabaseManager initialise"),
    ("11:49:14", "A1", "SUCCESS", "Normalize OK"),
    ("11:49:16", "A2", "SUCCESS", "Classification OK"),
    ("11:49:29", "A3", "SUCCESS", "Anomalies scored"),
    ("11:49:30", "A4", "ERROR", "google_search timeout"),
]


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Bienvenue. Chargez vos fichiers P&L, puis lancez une requete ici. "
                    "Le rapport et les graphes apparaissent a droite."
                ),
            }
        ]
    if "last_report" not in st.session_state:
        st.session_state.last_report = "Aucun rapport pour le moment."
    if "last_run_time" not in st.session_state:
        st.session_state.last_run_time = None
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []
    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = "idle"
    if "run_history" not in st.session_state:
        st.session_state.run_history = []
    if "agent_metrics" not in st.session_state:
        st.session_state.agent_metrics = AGENTS_DATA.copy()
    if "quality_metrics" not in st.session_state:
        st.session_state.quality_metrics = QUALITY_DATA.copy()
    if "operational_logs" not in st.session_state:
        st.session_state.operational_logs = STATIC_LOGS.copy()


def _resolve_google_api_key() -> str | None:
    """Resolve API key from Streamlit secrets first, then environment."""
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return str(st.secrets["GOOGLE_API_KEY"])
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY")


def execute_pipeline(query: str, uploaded_files: list) -> dict:
    """Run multi-agent execution via notebook backend."""
    key = _resolve_google_api_key()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Multi-agent execution requires a valid API key.")

    return execute_multi_agent_pipeline(
        query=query,
        uploaded_files=uploaded_files,
        api_key=key,
    )


def export_report_bytes(report_text: str) -> bytes:
    buffer = BytesIO()
    buffer.write(report_text.encode("utf-8"))
    return buffer.getvalue()


st.set_page_config(
    page_title="Enterprise Financial Performance Platform",
    page_icon="PL",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
      radial-gradient(circle at 12% 10%, rgba(20,184,166,0.12), transparent 30%),
      radial-gradient(circle at 88% 15%, rgba(59,130,246,0.14), transparent 32%),
      linear-gradient(165deg, #070b14 0%, #0b1220 100%);
}

[data-testid="stAppViewContainer"] {
    color: #f8fafc;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li,
[data-testid="stChatMessageContent"] p {
    color: #f8fafc !important;
}

[data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stAppViewContainer"] [data-testid="stChatMessage"],
[data-testid="stAppViewContainer"] [data-testid="stTextArea"],
[data-testid="stAppViewContainer"] [data-testid="stFileUploader"],
[data-testid="stAppViewContainer"] [data-testid="stMetric"] {
    background-color: #121a2b !important;
    border-color: #2a3449 !important;
}

[data-testid="stAppViewContainer"] .stTextArea textarea,
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] [data-baseweb="select"] {
    color: #f8fafc !important;
    background-color: #0f172a !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

.panel-title {
    font-weight: 700;
    letter-spacing: 0.2px;
    color: #f8fafc;
    margin-bottom: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Enterprise Financial Performance Platform")
st.caption("Professional workspace for financial analysis, anomaly intelligence, and executive decision support.")

with st.sidebar:
    page = st.radio("Navigation", NAV_ITEMS, index=0)
    st.markdown("---")
    st.markdown("### Chargement fichiers")
    uploaded_files = st.file_uploader(
        "Importer budget, actuals, mapping...",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls", "json", "txt"],
    )
    st.session_state.uploaded_file_names = [f.name for f in uploaded_files] if uploaded_files else []
    st.write(f"Fichiers en session: {len(st.session_state.uploaded_file_names)}")
    for name in st.session_state.uploaded_file_names:
        st.write(f"- {name}")

    st.markdown("---")
    st.markdown("### Etat pipeline")
    st.write(f"Statut: {st.session_state.pipeline_status}")
    if st.session_state.last_run_time:
        st.caption(f"Dernier run: {st.session_state.last_run_time}")

if page == "Executive Workspace":
    main_left, main_right = st.columns([1.15, 1], gap="large")

    with main_left:
        st.markdown('<div class="panel-title">Conversation & Strategic Requests</div>', unsafe_allow_html=True)
        with st.container(border=True, height=530):
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        prompt = st.chat_input("Enter a strategic request (e.g., analyze Q4 variances and propose an action plan).")

        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            st.session_state.pipeline_status = "running"
            try:
                with st.spinner("Running financial analysis pipeline..."):
                    result = execute_pipeline(prompt, uploaded_files or [])
            except Exception as exc:
                st.session_state.pipeline_status = "error"
                st.error(f"Pipeline execution failed: {exc}")
                assistant_msg = "Pipeline execution failed. Please verify API key and backend dependencies."
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
                with st.chat_message("assistant"):
                    st.markdown(assistant_msg)
                st.stop()

            st.session_state.last_report = result["report"]
            st.session_state.last_figures = result["figures"]
            st.session_state.last_kpis = result["kpis"]
            st.session_state.agent_metrics = result.get("agents", AGENTS_DATA.copy())
            st.session_state.quality_metrics = result.get("quality", QUALITY_DATA.copy())
            st.session_state.operational_logs = result.get("logs", STATIC_LOGS.copy())
            run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.last_run_time = run_time
            st.session_state.pipeline_status = "done"
            st.session_state.run_history.append(
                {
                    "time": run_time,
                    "query": prompt,
                    "files": len(uploaded_files or []),
                    "status": "done",
                }
            )

            assistant_msg = "Pipeline execution completed. Report and charts have been updated."
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
            with st.chat_message("assistant"):
                st.markdown(assistant_msg)

    with main_right:
        st.markdown('<div class="panel-title">Executive Report</div>', unsafe_allow_html=True)
        with st.container(border=True, height=250):
            st.text_area(
                "Generated report",
                value=st.session_state.last_report,
                height=190,
                label_visibility="collapsed",
            )

        st.download_button(
            "Download report (.txt)",
            data=export_report_bytes(st.session_state.last_report),
            file_name="pipeline_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("\n")
        st.markdown('<div class="panel-title">Charts</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if "last_kpis" in st.session_state:
                kpi_cols = st.columns(4)
                for idx, row in st.session_state.last_kpis.iterrows():
                    kpi_cols[idx].metric(row["Metric"], row["Value"])

            if "last_figures" in st.session_state:
                for fig in st.session_state.last_figures:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Charts will appear here after the first run.")

elif page == "Executive Dashboard":
    st.subheader("Pipeline Performance Overview")
    dashboard_agents = st.session_state.get("agent_metrics", AGENTS_DATA)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs", str(len(st.session_state.run_history)))
    c2.metric("Score A5", "7.2 / 10")
    c3.metric("Anomalies", "17")
    c4.metric("Status", st.session_state.pipeline_status)

    d1, d2 = st.columns(2)
    with d1:
        fig_success = px.bar(
            dashboard_agents,
            x="Agent",
            y="Succes",
            color="Statut",
            title="Agent Success Rate",
            color_discrete_map={"OK": "#16a34a", "ERROR": "#dc2626", "WAITING": "#f59e0b"},
        )
        st.plotly_chart(fig_success, use_container_width=True)
    with d2:
        fig_duration = px.line(dashboard_agents, x="Agent", y="Duree_s", markers=True, title="Average Duration (s)")
        st.plotly_chart(fig_duration, use_container_width=True)

elif page == "Run History":
    st.subheader("Recent Execution History")
    if st.session_state.run_history:
        history_df = pd.DataFrame(st.session_state.run_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No run has been executed yet.")

    st.markdown("### Latest report")
    st.code(st.session_state.last_report)

elif page == "Performance Insights":
    st.subheader("Performance and Quality Trends")
    quality_metrics = st.session_state.get("quality_metrics", QUALITY_DATA)
    p1, p2 = st.columns(2)
    with p1:
        fig_trend = px.line(
            quality_metrics,
            x="Date",
            y="Score Global",
            markers=True,
            title="Global Score Trend",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    with p2:
        fig_action = px.bar(quality_metrics, x="Date", y="Actionnabilite", title="Actionability")
        st.plotly_chart(fig_action, use_container_width=True)

elif page == "Operational Logs":
    st.subheader("Key Operational Logs")
    logs_df = pd.DataFrame(st.session_state.get("operational_logs", STATIC_LOGS), columns=["Time", "Agent", "Status", "Message"])
    st.dataframe(logs_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"Enterprise Financial Performance Platform v{APP_VERSION} | {datetime.now().strftime('%Y-%m-%d')}")
