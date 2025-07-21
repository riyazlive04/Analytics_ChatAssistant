# app.py

import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from google.cloud import bigquery
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
from datetime import datetime
import uuid
import json

# -----------------------
# Utility: Flatten GA4 event_params
# -----------------------
def flatten_event_params(df, field="event_params"):
    """Extracts key-value pairs from GA4's event_params into flat columns"""
    if field not in df.columns:
        return df

    expanded = {}
    for idx, row in df.iterrows():
        try:
            params = json.loads(row[field]) if isinstance(row[field], str) else row[field]
            if isinstance(params, list):
                for param in params:
                    param_name = param.get("key") or param.get("name")
                    val_obj = param.get("value") or {}
                    param_value = val_obj.get("stringValue") or val_obj.get("intValue") or val_obj.get("floatValue")
                    if param_name:
                        col = f"{field}_{param_name}"
                        if col not in expanded:
                            expanded[col] = [None] * len(df)
                        expanded[col][idx] = param_value
        except Exception:
            pass

    df = df.drop(columns=[field])
    for col, values in expanded.items():
        df[col] = values
    return df


# -----------------------
# Session ID & Logging
# -----------------------
if "user_session_id" not in st.session_state:
    st.session_state["user_session_id"] = str(uuid.uuid4())


def log_question_to_file(question):
    if not question.strip():
        return
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file = f"session-{st.session_state['user_session_id']}-{datetime.today().strftime('%Y%m%d')}.txt"
    with open(os.path.join(log_dir, file), "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%H:%M:%S")
        f.write(f"[{timestamp}] {question.strip()}\n")


def add_floating_followup_button():
    st.markdown("""
    <style>
    .float-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
    }
    .float-btn button {
        background-color: #2a9df4;
        color: white;
        padding: 12px 16px;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    </style>
    <div class="float-btn">
        <button onclick="document.getElementById('followup-box').scrollIntoView({behavior: 'smooth'});">
            💬 Ask Follow-up
        </button>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Environment & Clients
# -----------------------
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(__file__), "Keys", "service_account_key.json"
)
project_id = os.getenv("PROJECT_ID")

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
)
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
bq_client = bigquery.Client(project=project_id)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Smart Business Insight Assistant", layout="wide")
st.title("📊 Smart Business Insight Assistant")
st.markdown("Ask questions on BigQuery GA4 data and get instant business insights.")

# -----------------------
# Dataset/Table Selection
# -----------------------
datasets = [ds.dataset_id for ds in bq_client.list_datasets(project=project_id)]
dataset_id = st.selectbox("📁 Select Dataset", datasets)

if dataset_id:
    tables = [t.table_id for t in bq_client.list_tables(dataset_id)]
    table_id = st.selectbox("🗂️ Select Table", tables)

# -----------------------
# Optional Date Filter
# -----------------------
use_date_filter = st.checkbox("⏳ Filter by event_date?")
start_date = end_date = None
if use_date_filter:
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=datetime.today())
    with col2:
        end_date = st.date_input("End date", value=datetime.today())
    if start_date > end_date:
        st.warning("Start date must be before end date.")
        st.stop()

# -----------------------
# Load Data Button
# -----------------------
if table_id and st.button("🔎 Load Table"):
    full_id = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{full_id}`"
    if use_date_filter:
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        query += f" WHERE CAST(event_date AS STRING) BETWEEN '{start_str}' AND '{end_str}'"
        st.session_state["date_filter"] = (start_str, end_str)
    else:
        st.session_state["date_filter"] = None
    query += " LIMIT 1000"

    try:
        df = bq_client.query(query).result().to_dataframe()

        # Handle all objects/json properly
        for col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

        df = flatten_event_params(df, field="event_params")

        st.session_state["df"] = df
        st.session_state["data_loaded"] = True
        st.session_state["follow_qa"] = []
        st.session_state["last_question"] = ""
        st.session_state["insights"] = ""
        st.success("✅ Data loaded successfully!")
    except Exception as e:
        st.error(f"❌ Load failed: {e}")
        st.stop()

# -----------------------
# Display Table & Ask Question
# -----------------------
if st.session_state.get("data_loaded"):
    df = st.session_state["df"]
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 💬 Ask a Business Question")
    default_qs = [
        "What is the overall trend in the data?",
        "Which category performs the best?",
        "Are there seasonal trends?",
        "What drives most conversions?",
        "Summarize 3 actionable opportunities."
    ]
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_q = st.selectbox("Suggested question:", ["-- Select --"] + default_qs)
    with col2:
        custom_q = st.text_input("Or write your own:", key="custom_question_input")

    selected_q = "" if selected_q.startswith("--") else selected_q
    final_q = custom_q.strip() or selected_q.strip()

    if final_q and final_q != st.session_state.get("last_question"):
        st.session_state["last_question"] = final_q
        st.info("🔎 Generating answer, please wait...")

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_data = buf.getvalue()
        st.session_state["csv_data"] = csv_data

        log_question_to_file(final_q)

        prompt = f"""
You are a business analyst. Respond to the question below using this CSV data.
Give actionable insights, trends, and suggestions in markdown.

Data:
{csv_data[:4000]}

Question:
{final_q}
"""
        insights = chain.run(prompt)
        st.session_state["insights"] = insights

# -----------------------
# Show Insights
# -----------------------
if st.session_state.get("insights"):
    insights = st.session_state["insights"]
    df = st.session_state["df"]
    st.markdown("### ✅ Insights & Recommendations")
    st.markdown(insights, unsafe_allow_html=True)

    if "bar chart" in insights.lower() and df.shape[1] >= 2:
        try:
            fig, ax = plt.subplots()
            df.iloc[:, :2].groupby(df.columns[0]).sum().plot(kind="bar", ax=ax)
            st.pyplot(fig)
        except Exception as err:
            st.warning(f"⚠️ Chart issue: {err}")

    add_floating_followup_button()

# -----------------------
# Follow-up Q&A Section
# -----------------------
if st.session_state.get("insights"):
    st.markdown('<div id="followup-box"></div>', unsafe_allow_html=True)
    st.markdown("### 🔁 Ask a Follow-up")

    if "follow_qa" not in st.session_state:
        st.session_state["follow_qa"] = []

    follow_up = st.text_area(
        "Enter your follow-up question:", key=f"followup_input_{len(st.session_state['follow_qa'])}"
    )

    user_files = st.file_uploader("Upload optional files", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)

    if st.button("Submit Follow-up", key=f"submit_fqa_{len(st.session_state['follow_qa'])}"):
        st.info("🔎 Processing your follow-up...")

        files_info = "\n".join([f.name for f in user_files]) if user_files else "None"
        log_question_to_file(follow_up)

        full_prompt = f"""Previous Insights:\n{insights}

Follow-up Question:
{follow_up}

Context Files:
{files_info}
"""
        reply = chain.run(full_prompt)
        st.session_state["follow_qa"].append({"q": follow_up, "a": reply})
        st.rerun()

    for i, qa in enumerate(st.session_state["follow_qa"]):
        st.markdown(f"**Q{i+1}:** {qa['q']}")
        st.markdown(f"**A{i+1}:** {qa['a']}", unsafe_allow_html=True)

    # Export Options
    st.markdown("### 📤 Export Options")
    st.download_button("📄 Download CSV Table", data=st.session_state["csv_data"].encode(), file_name="insight_data.csv")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in insights.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    st.download_button("📄 Download Insights PDF", data=pdf_bytes, file_name="insights_summary.pdf")