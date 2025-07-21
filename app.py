# app.py

import streamlit as st
import os
import json
import uuid
import io
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
from google.cloud import bigquery
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


# -----------------------
# Set credentials from Streamlit Cloud Secrets — ✅ ADDED HERE
# -----------------------
if "SERVICE_ACCOUNT_KEY_JSON" in st.secrets:
    with open("gcp_key.json", "w") as f:
        f.write(st.secrets["SERVICE_ACCOUNT_KEY_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
    project_id = st.secrets["PROJECT_ID"]
else:
    st.error("❌ Google service account key is missing in Streamlit secrets.")
    st.stop()

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
    log_name = f"session-{st.session_state['user_session_id']}-{datetime.today().strftime('%Y%m%d')}.txt"
    path = os.path.join(log_dir, log_name)
    with open(path, "a", encoding="utf-8") as f:
        time = datetime.now().strftime('%H:%M:%S')
        f.write(f"[{time}] {question.strip()}\n")


# -----------------------
# Floating Button
# -----------------------
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
# LLM & BigQuery Clients
# -----------------------
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, api_key=st.secrets["OPENAI_API_KEY"])
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
bq_client = bigquery.Client(project=project_id)


# -----------------------
# Streamlit UI Setup
# -----------------------
st.set_page_config(page_title="📊 Smart Business Insight Assistant", layout="wide")
st.title("📊 Smart Business Insight Assistant")
st.markdown("Ask questions on BigQuery GA4 data and get instant business insights.")


# -----------------------
# Dataset & Table Selection
# -----------------------
datasets = [ds.dataset_id for ds in bq_client.list_datasets(project=project_id)]
dataset_id = st.selectbox("📁 Select a Dataset", datasets)

if dataset_id:
    tables = [t.table_id for t in bq_client.list_tables(dataset_id)]
    table_id = st.selectbox("🧾 Select a Table", tables)


# -----------------------
# Date Filter
# -----------------------
use_filter = st.checkbox("📅 Filter by event_date?")
start_date = end_date = None
if use_filter:
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=datetime.today())
    with col2:
        end_date = st.date_input("End date", value=datetime.today())
    if start_date > end_date:
        st.warning("Start date must be before End date.")
        st.stop()


# -----------------------
# Load Data Button
# -----------------------
if table_id and st.button("🔎 Load Table"):
    full_id = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{full_id}`"
    if use_filter:
        query += f" WHERE CAST(event_date AS STRING) BETWEEN '{start_date.strftime('%Y%m%d')}' AND '{end_date.strftime('%Y%m%d')}'"

    query += " LIMIT 1000"
    try:
        df = bq_client.query(query).result().to_dataframe()
        for col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        df = flatten_event_params(df)

        st.session_state["df"] = df
        st.session_state["data_loaded"] = True
        st.session_state["last_question"] = ""
        st.session_state["follow_qa"] = []
        st.session_state["insights"] = ""
        st.session_state["csv_data"] = df.to_csv(index=False)
        st.success("✅ Data loaded successfully!")

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()


# -----------------------
# Main Business Question
# -----------------------
if st.session_state.get("data_loaded"):
    df = st.session_state["df"]
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("### 💬 Ask a Business Question")

    suggested = [
        "What is the overall trend in the data?",
        "Which product category performs best?",
        "Are there seasonal trends in the data?",
        "What drives user engagement or conversions?",
        "Give me 3 key insights from this dataset."
    ]
    col1, col2 = st.columns((1, 2))
    with col1:
        selected_q = st.selectbox("Suggested Questions:", ['-- Select --'] + suggested)
    with col2:
        custom_q = st.text_input("Or write your question:")

    selected_q = "" if selected_q == "-- Select --" else selected_q
    final_q = custom_q.strip() or selected_q.strip()

    if final_q and final_q != st.session_state.get("last_question"):
        st.session_state["last_question"] = final_q
        st.info("🔎 Thinking through your question...")

        log_question_to_file(final_q)

        prompt = f"""
You are a strategic analyst. Answer the following business question using the provided dataset.
Be concise. Use bullet points. Highlight patterns and opportunities.

Data:
{st.session_state['csv_data'][:4000]}

Question:
{final_q}
"""
        insights = chain.run(prompt)
        st.session_state["insights"] = insights


# -----------------------
# Show Insights & Follow-up
# -----------------------
if st.session_state.get("insights"):
    insights = st.session_state["insights"]
    st.markdown("### ✅ Insights & Recommendations")
    st.markdown(insights, unsafe_allow_html=True)

    if "bar chart" in insights.lower() and df.shape[1] >= 2:
        try:
            fig, ax = plt.subplots()
            df.iloc[:, :2].groupby(df.columns[0]).sum().plot(kind="bar", ax=ax)
            st.pyplot(fig)
        except Exception as err:
            st.warning(f"⚠️ Unable to build chart: {err}")

    add_floating_followup_button()

    st.markdown('<div id="followup-box"></div>', unsafe_allow_html=True)
    st.markdown("### 🔁 Ask a Follow-up")

    follow_up = st.text_area("Your next question:", key=f"fup_{len(st.session_state['follow_qa'])}")
    uploaded_files = st.file_uploader("Attach files (optional)", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)

    if st.button("Submit Follow-up", key=f"btn_{len(st.session_state['follow_qa'])}"):
        st.info("🔎 Processing follow-up...")
        log_question_to_file(follow_up)
        file_list = ", ".join([f.name for f in uploaded_files]) if uploaded_files else "None"

        prompt_follow = f"""
Context:
{insights}

Follow-up Question:
{follow_up}

Files attached (names only):
{file_list}
"""
        reply = chain.run(prompt_follow)
        st.session_state["follow_qa"].append({"q": follow_up, "a": reply})
        st.rerun()

    # Display all previous QA
    for i, turn in enumerate(st.session_state["follow_qa"]):
        st.markdown(f"**🔁 Follow-up Q{i+1}:** {turn['q']}")
        st.markdown(f"**🧠 Answer:** {turn['a']}", unsafe_allow_html=True)

    # Export options
    st.markdown("### 📦 Export")
    st.download_button("⬇️ Download CSV", data=st.session_state["csv_data"].encode(), file_name="data.csv")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in insights.split("\n"):
        pdf.multi_cell(0, 10, line)
    st.download_button("⬇️ Download PDF", data=pdf.output(dest="S").encode("latin-1"), file_name="insights.pdf")
