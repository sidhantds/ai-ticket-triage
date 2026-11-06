import re
import sqlite3
from pathlib import Path

import joblib
import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="AI Ticket Operations",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = str(BASE_DIR / "tickets.db")
MODEL_VERSION = "v1.0-logreg-tfidf"
CONFIDENCE_THRESHOLD = 0.40
SLA_HOURS = {"high": 1, "medium": 8, "low": 24}
MIN_TICKET_LENGTH = 10
MAX_TICKET_LENGTH = 10_000


@st.cache_resource
def load_models():
    return (
        joblib.load(BASE_DIR / "tfidf_vectorizer.pkl"),
        joblib.load(BASE_DIR / "category_model.pkl"),
        joblib.load(BASE_DIR / "priority_model.pkl"),
    )


@st.cache_resource
def load_nlp_tools():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    words = set(stopwords.words("english"))
    words.difference_update({"not", "no"})
    words.update({
        "dear", "regards", "hello", "hi", "thank", "thanks",
        "sincerely", "kind", "best", "name", "us", "could",
        "would", "please",
    })
    return words, WordNetLemmatizer()


tfidf, category_model, priority_model = load_models()
stop_words, lemmatizer = load_nlp_tools()
feature_names = tfidf.get_feature_names_out()


def redact_pii(text: str) -> str:
    text = re.sub(r"(?<![\w.-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
                  "[EMAIL_REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b",
        "[PHONE_REDACTED]",
        text,
    )
    text = re.sub(r"\b(?:\+?1)?\d{10,14}\b", "[PHONE_REDACTED]", text)
    text = re.sub(r"\b\d{6,}\b", "[ID_REDACTED]", text)
    return text


def validate_ticket(subject: str, body: str) -> str:
    if not body or not body.strip():
        raise ValueError("Ticket body is required.")

    combined = f"{subject or ''} {body}".strip()
    if len(combined) > MAX_TICKET_LENGTH:
        raise ValueError(
            f"Ticket text exceeds the {MAX_TICKET_LENGTH}-character limit."
        )
    if len(body.strip()) < MIN_TICKET_LENGTH:
        raise ValueError(
            f"Ticket body must contain at least {MIN_TICKET_LENGTH} characters."
        )

    return redact_pii(combined)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 1
    )


def analyze_ticket(text):
    vector = tfidf.transform([clean_text(text)])
    category_probs = category_model.predict_proba(vector)[0]
    priority_probs = priority_model.predict_proba(vector)[0]
    category_index = int(category_probs.argmax())
    priority_index = int(priority_probs.argmax())
    category = str(category_model.classes_[category_index])
    priority = str(priority_model.classes_[priority_index]).strip().lower()
    category_confidence = round(float(category_probs[category_index]) * 100, 1)
    priority_confidence = round(float(priority_probs[priority_index]) * 100, 1)

    top_categories = [
        (str(category_model.classes_[index]), round(float(category_probs[index]) * 100, 1))
        for index in category_probs.argsort()[::-1][:3]
    ]

    coefficients = category_model.coef_
    if coefficients.ndim == 1:
        selected_coefficients = coefficients if category_index == 1 else -coefficients
    else:
        selected_coefficients = coefficients[category_index]
    contributions = vector.toarray()[0] * selected_coefficients
    key_terms = [
        str(feature_names[index])
        for index in contributions.argsort()[::-1][:5]
        if contributions[index] > 0
    ]

    return {
        "category": category,
        "category_confidence": category_confidence,
        "priority": priority,
        "priority_confidence": priority_confidence,
        "sla_hours": SLA_HOURS.get(priority, 24),
        "status": "Auto-Routed" if category_confidence >= CONFIDENCE_THRESHOLD * 100 else "Needs Human Review",
        "key_terms": key_terms,
        "top_categories": top_categories,
    }


def get_connection():
    return sqlite3.connect(DB_FILE, timeout=30)


def create_tables():
    with get_connection() as connection:
        connection.executescript("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            body TEXT NOT NULL,
            actual_category TEXT,
            actual_priority TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER NOT NULL,
            predicted_category TEXT,
            category_confidence REAL,
            predicted_priority TEXT,
            priority_confidence REAL,
            sla_hours INTEGER,
            status TEXT,
            key_words TEXT,
            model_version TEXT,
            FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
        );
        """)


@st.cache_data(ttl=5)
def run_query(sql):
    with get_connection() as connection:
        return pd.read_sql_query(sql, connection)


def save_ticket(subject, body, result):
    subject = redact_pii(str(subject or "")).strip()
    body = redact_pii(str(body or "")).strip()
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO tickets (subject, body) VALUES (?, ?)",
            (subject, body),
        )
        ticket_id = cursor.lastrowid
        connection.execute(
            """INSERT INTO predictions
               (ticket_id, predicted_category, category_confidence, predicted_priority,
                priority_confidence, sla_hours, status, key_words, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticket_id,
                result["category"],
                result["category_confidence"],
                result["priority"],
                result.get("priority_confidence"),
                result["sla_hours"],
                result["status"],
                ", ".join(result.get("key_terms", [])),
                MODEL_VERSION,
            ),
        )
    run_query.clear()
    return ticket_id


create_tables()

if "triage_result" not in st.session_state:
    st.session_state["triage_result"] = None

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {
    background-color: #0B1F3A;
    border-right: 1px solid #172B4D;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #FFFFFF;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(203, 213, 225, 0.18);
}
[data-testid="stMetric"] {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 16px;
    background: #FFFFFF;
}
.stTabs [data-baseweb="tab-list"] {gap: 12px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    page = option_menu(
        menu_title="Ticket AI",
        options=[
            "Overview",
            "Ticket Triage",
            "Bulk Classification",
            "Ticket Register",
            "Operations Dashboard",
            "Model Card",
        ],
        icons=[
            "grid",
            "ticket-perforated",
            "cloud-upload",
            "list-ul",
            "bar-chart",
            "journal-text",
        ],
        menu_icon="menu-app",
        default_index=0,
        key="main_navigation",
        styles={
            "container": {
                "padding": "0.5rem 0 0 0",
                "background-color": "#0B1F3A",
            },
            "menu-title": {
                "font-size": "18px",
                "font-weight": "700",
                "color": "#FFFFFF",
                "padding": "0 0.75rem 0.75rem 0.75rem",
            },
            "menu-icon": {
                "color": "#FFFFFF",
                "font-size": "18px",
            },
            "icon": {
                "color": "#FFFFFF",
                "font-size": "16px",
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px 0",
                "padding": "0.65rem 0.75rem",
                "border-radius": "6px",
                "--hover-color": "#172B4D",
                "color": "#FFFFFF",
            },
            "nav-link-selected": {
                "background-color": "#1E3A5F",
                "color": "#FFFFFF",
                "font-weight": "600",
            },
        },
    )
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {MODEL_VERSION}")
st.sidebar.caption(f"Human review below {int(CONFIDENCE_THRESHOLD * 100)}% confidence")


if page == "Overview":
    st.title("AI Ticket Operations")
    st.caption(
        "AI-assisted triage that classifies support tickets, recommends handling priority, "
        "and flags low-confidence cases for human review."
    )

    stats = run_query("""
        SELECT COUNT(*) AS total,
               AVG(category_confidence) AS confidence,
               SUM(CASE WHEN status = 'Auto-Routed' THEN 1 ELSE 0 END) AS auto_routed
        FROM predictions
    """)
    total = int(stats["total"][0])
    confidence = float(stats["confidence"][0] or 0)
    high = int(run_query(
        "SELECT COUNT(*) AS total FROM predictions WHERE predicted_priority = 'high'"
    )["total"][0])
    review = int(run_query(
        "SELECT COUNT(*) AS total FROM predictions WHERE status = 'Needs Human Review'"
    )["total"][0])
    auto_routed = int(stats["auto_routed"][0] or 0)
    automation_rate = (auto_routed / total * 100) if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tickets processed", total, help="Total tickets analyzed and stored by the system.")
    c2.metric("High priority", high)
    c3.metric("Human review", review)
    c4.metric("Automation rate", f"{automation_rate:.1f}%")
    c5.metric("Avg. confidence", f"{confidence:.1f}%", help="Average predicted category confidence.")

    overview_data = run_query("""
        SELECT t.created_at,
               p.predicted_priority,
               p.status
        FROM tickets t
        JOIN predictions p ON t.ticket_id = p.ticket_id
    """)

    left, right = st.columns(2)
    with left:
        st.subheader("Tickets processed")
        if overview_data.empty:
            st.caption("No ticket activity yet.")
        else:
            trend = overview_data.copy()
            trend["date"] = pd.to_datetime(trend["created_at"]).dt.date
            trend = trend.groupby("date").size().reset_index(name="tickets")
            fig = px.area(trend, x="date", y="tickets", color_discrete_sequence=["#6B7280"])
            fig.update_layout(
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                xaxis_title="",
                yaxis_title="Tickets",
                hovermode="x",
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Workload by priority")
        if overview_data.empty:
            st.caption("No priority data yet.")
        else:
            priority_counts = (
                overview_data.groupby("predicted_priority")
                .size()
                .reset_index(name="tickets")
            )
            fig = px.bar(
                priority_counts,
                x="tickets",
                y="predicted_priority",
                orientation="h",
                text="tickets",
                color="predicted_priority",
                category_orders={"predicted_priority": ["high", "medium", "low"]},
                color_discrete_map={
                    "high": "#DC2626",
                    "medium": "#F59E0B",
                    "low": "#6B7280",
                },
            )
            fig.update_layout(
                showlegend=False,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                xaxis_title="Tickets",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("System status")
    status_left, status_right = st.columns(2)
    with status_left:
        st.markdown(
            f"**Model version:** {MODEL_VERSION}<br>"
            "**Supported language:** English<br>"
            "**Prediction targets:** Ticket category and priority<br>"
            f"**Human-review threshold:** Below {int(CONFIDENCE_THRESHOLD * 100)}% category confidence",
            unsafe_allow_html=True,
        )
    with status_right:
        st.markdown(
            "**SLA recommendations**<br>"
            "- High priority: 1 hour<br>"
            "- Medium priority: 8 hours<br>"
            "- Low priority: 24 hours",
            unsafe_allow_html=True,
        )

    st.subheader("Quick actions")
    st.info(
        "Use **Ticket Triage** to analyze a new ticket, "
        "**Bulk Classification** to process a CSV, or "
        "**Operations Dashboard** for detailed filters and charts."
    )


elif page == "Ticket Triage":
    st.title("AI Ticket Triage")
    st.caption("Classify a customer issue and recommend its handling priority.")

    with st.form("ticket_form", clear_on_submit=False):
        st.subheader("Ticket details")
        subject = st.text_input("Subject", max_chars=200, key="ticket_subject")
        body = st.text_area(
            "Customer message",
            height=220,
            max_chars=10000,
            placeholder="Describe the customer issue...",
            key="ticket_body",
        )
        st.caption(
            "Current model supports English tickets. "
            "No customer-tier or language routing rules are applied in this release."
        )
        submitted = st.form_submit_button("Analyze ticket", type="primary", use_container_width=True)

    if submitted:
        try:
            sanitized_text = validate_ticket(subject, body)
            sanitized_subject = redact_pii(subject.strip())
            sanitized_body = redact_pii(body.strip())
            result = analyze_ticket(sanitized_text)
            result["ticket_id"] = save_ticket(sanitized_subject, sanitized_body, result)
            st.session_state["triage_result"] = result
        except ValueError as error:
            st.warning(str(error))

    result = st.session_state.get("triage_result")
    if result:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended category", result["category"].title())
        c2.metric("Recommended priority", result["priority"].title())
        c3.metric("Recommended SLA", f"{result['sla_hours']} hour(s)")

        tab1, tab2 = st.tabs(["Prediction", "Alternative categories"])
        with tab1:
            st.progress(
                result["category_confidence"] / 100,
                text=f"Category confidence: {result['category_confidence']:.1f}%",
            )
            if result["key_terms"]:
                st.write("Top signals: " + ", ".join(result["key_terms"]))
            else:
                st.write("No strong key terms were identified.")
            if result["status"] == "Needs Human Review":
                st.warning("Human review is recommended due to low category confidence.")
            else:
                st.success("The prediction meets the configured routing threshold.")
        with tab2:
            alternatives = pd.DataFrame(result["top_categories"], columns=["Category", "Confidence"])
            st.dataframe(alternatives, hide_index=True, use_container_width=True)
        st.caption(f"Saved as ticket #{result['ticket_id']}")
    else:
        st.info("Enter a ticket to generate a triage recommendation.")


elif page == "Bulk Classification":
    st.title("Bulk Classification")
    st.caption("Upload tickets, review predictions, and save approved results.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("The CSV must contain a body column and may include a subject column.")
    else:
        try:
            data = pd.read_csv(uploaded)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            st.error("This file could not be read as a valid CSV.")
            st.stop()
        if "body" not in data.columns:
            st.error("The CSV must contain a body column.")
        elif data.empty:
            st.info("The uploaded CSV contains no ticket rows.")
        else:
            data["subject"] = data["subject"].fillna("") if "subject" in data.columns else ""
            data["body"] = data["body"].fillna("")
            st.subheader("Upload preview")
            preview = data.head(10).copy()
            preview["subject"] = preview["subject"].map(redact_pii)
            preview["body"] = preview["body"].map(redact_pii)
            st.dataframe(preview, hide_index=True, use_container_width=True)
            if st.button("Analyze uploaded tickets", type="primary"):
                rows = []
                progress = st.progress(0.0)
                invalid_count = 0
                for index, (_, row) in enumerate(data.iterrows(), start=1):
                    try:
                        sanitized_text = validate_ticket(str(row["subject"]), str(row["body"]))
                    except ValueError:
                        invalid_count += 1
                        progress.progress(index / len(data))
                        continue
                    prediction = analyze_ticket(sanitized_text)
                    rows.append({
                        "subject": redact_pii(str(row["subject"])),
                        "body": redact_pii(str(row["body"])),
                        "category": prediction["category"],
                        "category_confidence": prediction["category_confidence"],
                        "priority": prediction["priority"],
                        "sla_hours": prediction["sla_hours"],
                        "status": prediction["status"],
                    })
                    progress.progress(index / len(data))
                if rows:
                    st.session_state["bulk_results"] = pd.DataFrame(rows)
                else:
                    st.session_state["bulk_results"] = None
                if invalid_count:
                    st.warning(
                        f"{invalid_count} row(s) were skipped because they were empty, "
                        "too short, or exceeded the character limit."
                    )

            reviewed = st.session_state.get("bulk_results")
            if reviewed is not None:
                st.subheader("Review queue")
                edited = st.data_editor(
                    reviewed,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "category": st.column_config.SelectboxColumn(
                            "Category", options=sorted(reviewed["category"].unique())
                        ),
                        "priority": st.column_config.SelectboxColumn(
                            "Priority", options=["high", "medium", "low"]
                        ),
                        "status": st.column_config.SelectboxColumn(
                            "Review status", options=["Auto-Routed", "Needs Human Review"]
                        ),
                        "category_confidence": st.column_config.ProgressColumn(
                            "Confidence", min_value=0, max_value=100, format="%.1f%%"
                        ),
                        "body": None,
                    },
                    key="bulk_editor",
                )
                c1, c2 = st.columns(2)
                if c1.button("Save reviewed rows", type="primary", use_container_width=True):
                    for _, row in edited.iterrows():
                        save_ticket(
                            str(row["subject"]),
                            str(row["body"]),
                            {
                                "category": row["category"],
                                "category_confidence": row["category_confidence"],
                                "priority": row["priority"],
                                "priority_confidence": None,
                                "sla_hours": row["sla_hours"],
                                "status": row["status"],
                                "key_terms": [],
                            },
                        )
                    st.success(f"Saved {len(edited)} reviewed rows.")
                c2.download_button(
                    "Download enriched CSV",
                    edited.to_csv(index=False),
                    file_name="classified_tickets.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


elif page == "Ticket Register":
    st.title("Ticket Register")
    st.caption("Search, filter, sort, and export classified tickets.")
    history = run_query("""
        SELECT t.ticket_id, t.created_at, t.subject, t.body,
               p.predicted_category AS category,
               p.predicted_priority AS priority,
               p.category_confidence AS confidence,
               p.status, p.sla_hours
        FROM tickets t
        JOIN predictions p ON t.ticket_id = p.ticket_id
        ORDER BY t.ticket_id DESC
    """)
    if history.empty:
        st.info("No classified tickets are available.")
    else:
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        search = c1.text_input("Search subject, body, or ticket ID", key="register_search")
        priorities = c2.multiselect("Priority", ["high", "medium", "low"], key="register_priority")
        statuses = c3.multiselect(
            "Status", ["Auto-Routed", "Needs Human Review"], key="register_status"
        )
        sort_by = c4.selectbox(
            "Sort by",
            ["Newest first", "Oldest first", "Highest confidence", "Lowest confidence", "Ticket ID"],
            key="register_sort",
        )

        filtered = history.copy()
        if search.strip():
            search_mask = (
                history["subject"].fillna("").str.contains(search, case=False, na=False)
                | history["body"].fillna("").str.contains(search, case=False, na=False)
                | history["ticket_id"].astype(str).str.contains(search, na=False)
            )
            filtered = filtered[search_mask]
        if priorities:
            filtered = filtered[filtered["priority"].isin(priorities)]
        if statuses:
            filtered = filtered[filtered["status"].isin(statuses)]

        if sort_by == "Newest first":
            filtered = filtered.sort_values("ticket_id", ascending=False)
        elif sort_by == "Oldest first":
            filtered = filtered.sort_values("ticket_id", ascending=True)
        elif sort_by == "Highest confidence":
            filtered = filtered.sort_values("confidence", ascending=False)
        elif sort_by == "Lowest confidence":
            filtered = filtered.sort_values("confidence", ascending=True)
        else:
            filtered = filtered.sort_values("ticket_id", ascending=True)

        st.caption(f"{len(filtered)} matching tickets")
        st.dataframe(
            filtered,
            hide_index=True,
            use_container_width=True,
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%.1f%%"
                ),
                "body": None,
            },
        )
        st.download_button(
            "Download filtered CSV",
            filtered.to_csv(index=False),
            file_name="ticket_register.csv",
            mime="text/csv",
        )


elif page == "Operations Dashboard":
    st.title("Operations Dashboard")
    st.caption("Filter ticket volume, priority, and model review workload.")
    data = run_query("""
        SELECT t.ticket_id, t.created_at,
               p.predicted_category, p.category_confidence,
               p.predicted_priority, p.status
        FROM tickets t JOIN predictions p ON t.ticket_id = p.ticket_id
    """)
    if data.empty:
        st.info("No operational data is available yet.")
    else:
        c1, c2, c3 = st.columns(3)
        categories = c1.multiselect(
            "Category",
            sorted(data["predicted_category"].dropna().unique()),
            key="dashboard_category",
        )
        priorities = c2.multiselect(
            "Priority", ["high", "medium", "low"], key="dashboard_priority"
        )
        review_only = c3.toggle("Human review only", key="dashboard_review")
        filtered = data.copy()
        if categories:
            filtered = filtered[filtered["predicted_category"].isin(categories)]
        if priorities:
            filtered = filtered[filtered["predicted_priority"].isin(priorities)]
        if review_only:
            filtered = filtered[filtered["status"] == "Needs Human Review"]

        total = len(filtered)
        high = int((filtered["predicted_priority"] == "high").sum())
        review = int((filtered["status"] == "Needs Human Review").sum())
        confidence = float(filtered["category_confidence"].mean() if total else 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickets", total)
        c2.metric("High priority", high)
        c3.metric("Automation rate", f"{(total - review) / total * 100:.1f}%" if total else "0%")
        c4.metric("Average confidence", f"{confidence:.1f}%")

        category_counts = filtered.groupby("predicted_category").size().reset_index(name="tickets")
        priority_counts = filtered.groupby("predicted_priority").size().reset_index(name="tickets")
        left, right = st.columns(2)
        with left:
            st.subheader("Tickets by category")
            fig = px.bar(category_counts, x="predicted_category", y="tickets", text="tickets",
                         color_discrete_sequence=["#4B5563"])
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Tickets",
                              margin={"l": 20, "r": 20, "t": 20, "b": 20})
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Tickets by priority")
            fig = px.pie(priority_counts, names="predicted_priority", values="tickets",
                         hole=0.55, color_discrete_sequence=["#374151", "#6B7280", "#D1D5DB"])
            fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
            st.plotly_chart(fig, use_container_width=True)


elif page == "Model Card":
    st.title("Model Card")
    st.caption("Transparency, limitations, and responsible use of the triage model.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Purpose")
        st.write("Assist support teams by suggesting ticket category and priority.")
        st.subheader("Intended users")
        st.write("Support agents, team leads, and operations managers.")
    with c2:
        st.subheader("Inputs")
        st.write("Ticket subject and body text.")
        st.subheader("Data")
        st.write("1,390 English tickets from the Kaggle multilingual support-ticket dataset.")
    with c3:
        st.subheader("Outputs")
        st.write("Category, priority, confidence, SLA recommendation, and key terms.")
        st.subheader("Human oversight")
        st.write("Low-confidence predictions are flagged for manual review.")

    st.subheader("Model comparison")
    category_comparison = BASE_DIR / "model_comparison_category.csv"
    priority_comparison = BASE_DIR / "model_comparison_priority.csv"
    left, right = st.columns(2)
    with left:
        st.caption("Category")
        if category_comparison.exists():
            st.dataframe(pd.read_csv(category_comparison), hide_index=True, use_container_width=True)
    with right:
        st.caption("Priority")
        if priority_comparison.exists():
            st.dataframe(pd.read_csv(priority_comparison), hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Known limitations")
        st.write("- English-only in the current release\n- Overlapping support categories\n- Public data may not reflect current operations\n- Priority labels may not match real business urgency")
    with right:
        st.subheader("Future roadmap")
        st.write("- Multilingual support\n- Real ticket metadata and active learning\n- Authentication and role-based access\n- ITSM integration and drift monitoring")

    st.subheader("PII and data-handling policy")
    st.write(
        "This prototype rejects empty, short, and over-limit ticket text. "
        "Email addresses, phone numbers, and long numeric identifiers are masked "
        "before prediction and database storage. Raw ticket bodies are not written "
        "to application logs. This is a prototype control, not a guarantee that "
        "all personal or sensitive information has been identified."
    )
