# spendly_app_v3_aesthetic.py
# Spendly — Chat-first, Explainable, Data-first
# Complete Aesthetic Overhaul: CSS consolidated for improved maintainability.

import os
import traceback
import calendar
from datetime import datetime
from typing import Optional, Dict
import json
import re

import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import plotly.io as pio
import streamlit.components.v1 as components
from urllib.parse import quote_plus

# -------- Try to import llm_client and responder (graceful fallback) ------
LLM_AVAILABLE = False
RESPONDER_AVAILABLE = False
try:
    from llm_client import llm_parse_query_xml, llm_available, simple_chat
    LLM_AVAILABLE = llm_available()
except Exception:
    def llm_parse_query_xml(q, known_categories=None):
        return None, "LLM unavailable."
    def simple_chat(messages, model="gpt-3.5-turbo", max_tokens=240, temperature=0.0):
        last_user = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), "")
        return last_user
    LLM_AVAILABLE = False
try:
    import responder
    if not hasattr(responder, "respond_to_parsed_intent") or not hasattr(responder, "init_db_from_df"):
        raise ImportError("responder missing required funcs")
    RESPONDER_AVAILABLE = True
except Exception:
    RESPONDER_AVAILABLE = False

# Page config
st.set_page_config(page_title="Spendly — Chat", layout="wide")
DATA_DIR = "data"
DEFAULT_XLSX = os.path.join(DATA_DIR, "Open Bank Transaction Data.xlsx")
DEFAULT_SHEET = "Anonymized Original with Catego"
DB_PATH = ":memory:"

# -------------------------------------------------------------------------
# --- CONSOLIDATED AESTHETIC/THEMING CSS OVERHAUL (Complete Redesign) ---
# -------------------------------------------------------------------------

# --- 1. Style Variables & Theme (Single source of truth) ---
PRIMARY_COLOR = "#5a57a8" # Spendly Primary Purple
SECONDARY_COLOR = "#4b459e"
TEXT_DARK = "#1a1a2e"
BG_SOFT = "#f7f9fc"

CSS_VARS = {
    "PRIMARY_COLOR": PRIMARY_COLOR,
    "SECONDARY_COLOR": SECONDARY_COLOR,
    "TEXT_DARK": TEXT_DARK,
    "BG_SOFT": BG_SOFT,
    "SHADOW_SOFT": "0 6px 15px rgba(20, 20, 50, 0.05)",
    "BORDER_SOFT": "1px solid rgba(0, 0, 0, 0.08)",
    "CHAT_BOX_SHADOW": "0 8px 20px rgba(20,20,50,0.04)",
    "CHAT_BOX_BORDER": "1px solid rgba(0,0,0,0.06)",
    "USER_BUBBLE_SHADOW": "0 6px 15px rgba(75,69,158,0.2)",
}

# --- 2. GLOBAL APP CSS (Applied via st.markdown) ---
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Streamlit Overrides */
html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    color: {CSS_VARS['TEXT_DARK']}; 
    font-size: 16px; 
}}
[data-testid="stAppViewContainer"] > .main {{
    background-color: {CSS_VARS['BG_SOFT']}; 
}}
.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}}

/* Typography */
strong, b, .stMarkdown strong, .stMarkdown b {{ font-weight: 700 !important; }}
h1, h2, h3, h4 {{ font-weight: 600 !important; color: {CSS_VARS['TEXT_DARK']}; }}

/* Enhancing Metrics (KPIs) */
[data-testid="stMetric"] {{
    background-color: #ffffff;
    border: {CSS_VARS['BORDER_SOFT']};
    border-radius: 12px;
    padding: 15px 20px;
    box-shadow: {CSS_VARS['SHADOW_SOFT']}; 
    transition: all 0.2s ease-in-out;
}}
[data-testid="stMetricLabel"] {{
    color: #555 !important;
    font-size: 14px;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: {CSS_VARS['SECONDARY_COLOR']} !important; 
}}

/* File Uploader styling */
[data-testid="stFileUploader"] > div {{
    border-radius: 10px;
    border: 1px dashed {CSS_VARS['PRIMARY_COLOR']};
    background-color: #ffffff;
}}

/* Form Button styling */
[data-testid="stForm"] button[type="submit"] {{
    border-radius: 8px;
    font-weight: 600;
}}
</style>
"""

# --- 3. CHAT IFRAME CSS (Applied via components.html) ---
def get_chat_iframe_css():
    return f"""
    <style>
      /* CRITICAL FIX: Ensure HTML/Body fill the iframe */
      html, body {{ 
        margin:0; 
        padding:0; 
        height:100%; 
        -webkit-font-smoothing:antialiased; 
        -moz-osx-font-smoothing:grayscale; 
      }}
      body {{ font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background:transparent; color: {CSS_VARS['TEXT_DARK']}; }}
      
      /* New: Ensure the outer div fills its parent (the body) and uses flex for height distribution */
      div:first-of-type {{ 
        height: 100%;
        display: flex;
        flex-direction: column;
      }}

      /* Chat History Container Styling */
      .chat-history-container {{ 
        flex-grow: 1; 
        height: 100%; 
        max-height: 100%; 
        overflow-y:auto !important; 
        -webkit-overflow-scrolling: touch; 
        padding:16px; 
        box-sizing:border-box; 
        margin-bottom: 2px; 
        background: #ffffff; 
        border-radius:16px; 
        border:{CSS_VARS['CHAT_BOX_BORDER']}; 
        box-shadow: {CSS_VARS['CHAT_BOX_SHADOW']}; 
      }}
      .chat-history-container::-webkit-scrollbar{{width:8px}}
      .chat-history-container::-webkit-scrollbar-thumb{{background:rgba(0,0,0,0.1);border-radius:8px}}
      
      /* Chat Bubble Styling (REDUCED BUBBLE SIZE AND LINE DENSITY) */
      .chat-bubble{{
        width: fit-content; /* Critical: Shrink-wrap the bubble width */
        padding: 8px 12px; 
        margin: 6px 0; 
        max-width:80%;
        border-radius:14px; 
        white-space:pre-wrap;
        word-break:break-word;
        line-height: 1.4; 
        font-size: 15px; 
      }}
      /* Assistant Bubble */
      .chat-bubble.assistant{{
        background:linear-gradient(180deg,#ffffff 0%,#fbfcff 100%);
        border:1px solid rgba(115,103,240,0.08);
        color:{CSS_VARS['TEXT_DARK']}; 
        border-bottom-left-radius: 4px;
      }}
      /* User Bubble */
      .chat-bubble.user{{
        background:linear-gradient(180deg,{CSS_VARS['PRIMARY_COLOR']},#4b459e);
        color:#fff;
        margin-left:auto;
        box-shadow:{CSS_VARS['USER_BUBBLE_SHADOW']}; 
        border-bottom-right-radius: 4px;
      }}
      
      strong, b {{ font-weight:700 !important; }}
      pre {{ 
        white-space:pre-wrap; 
        word-break:break-word; 
        margin-top:8px; 
        max-height:40vh; 
        overflow:auto; 
        background:#f7f7fb; 
        padding:10px; 
        border-radius:8px; 
        border: 1px solid rgba(0,0,0,0.05); 
      }}
      details summary::marker {{ content: none; }}
      hr {{ border-top: 1px solid rgba(0,0,0,0.08); margin: 12px 0; }}
    </style>
    """

# Apply the Global CSS block
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# --- HELPER FUNCTIONS (Unchanged from your submission, omitted for brevity) ---
# -------------------------------------------------------------------------

def local_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # ... (function body omitted for brevity, assumed to be correct)
    d = df.copy()
    date_col = None
    for c in d.columns:
        if 'date' in c.lower():
            date_col = c
            break
    if date_col:
        d['Transaction_Date'] = pd.to_datetime(d[date_col], dayfirst=True, errors='coerce')
    else:
        if 'Transaction Date' in d.columns:
            d['Transaction_Date'] = pd.to_datetime(d['Transaction Date'], dayfirst=True, errors='coerce')
        else:
            d['Transaction_Date'] = pd.NaT
    d['Debit_Amount'] = pd.to_numeric(d.get('Debit Amount', d.get('DebitAmount', d.get('Debit', 0))), errors='coerce').fillna(0.0)
    d['Credit_Amount'] = pd.to_numeric(d.get('Credit Amount', d.get('CreditAmount', d.get('Credit', 0))), errors='coerce').fillna(0.0)
    d['Amount'] = d['Debit_Amount'] - d['Credit_Amount']
    d['Txn_Number'] = d.get('Transaction Number', pd.Series(range(1, len(d)+1)))
    d['Txn_Type'] = d.get('Transaction Type', '').astype(str) if 'Transaction Type' in d.columns else ''
    d['Description'] = d.get('Transaction Description', d.get('Description', '')).astype(str)
    d['Balance'] = pd.to_numeric(d.get('Balance', np.nan), errors='coerce')
    d['Category'] = d.get('Category', 'Uncategorized').fillna('Uncategorized').astype(str)
    d['City'] = d.get('Location City', d.get('City', '')).astype(str)
    d['Country'] = d.get('Location Country', d.get('Country', '')).astype(str)
    d = d.dropna(subset=['Transaction_Date'])
    d['Year'] = d['Transaction_Date'].dt.year
    d['Month'] = d['Transaction_Date'].dt.month
    d['YearMonth'] = d['Transaction_Date'].dt.to_period('M').astype(str)
    d['Week'] = d['Transaction_Date'].dt.to_period('W').astype(str)
    d['Is_Spend'] = d['Amount'] > 0
    d['Is_Income'] = d['Credit_Amount'] > 0
    return d

def local_init_db_from_df(df: pd.DataFrame, db_path=":memory:"):
    # ... (function body omitted for brevity, assumed to be correct)
    con = duckdb.connect(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS transactions;")
    except:
        pass
    con.register("mem_df", df)
    create_sql = """
    CREATE TABLE transactions AS
    SELECT
        COALESCE(mem_df.Txn_Number, ROW_NUMBER() OVER()) AS Txn_Number,
        mem_df.Transaction_Date,
        mem_df.Txn_Type,
        mem_df.Description,
        COALESCE(mem_df.Debit_Amount,0) AS Debit_Amount,
        COALESCE(mem_df.Credit_Amount,0) AS Credit_Amount,
        COALESCE(mem_df.Amount,0) AS Amount,
        mem_df.Balance,
        mem_df.Category,
        mem_df.City,
        mem_df.Country,
        mem_df.Year,
        mem_df.Month,
        mem_df.YearMonth,
        mem_df.Week,
        COALESCE(mem_df.Is_Spend, false) AS Is_Spend,
        COALESCE(mem_df.Is_Income, false) AS Is_Income
    FROM mem_df;
    """
    try:
        con.execute(create_sql)
    except:
        con.execute("CREATE TABLE transactions AS SELECT * FROM mem_df;")
    con.unregister("mem_df")
    return con

def local_sql_top_categories(con, n=10):
    q = f"SELECT Category, SUM(Amount) AS spend FROM transactions WHERE Is_Spend = TRUE GROUP BY Category ORDER BY spend DESC LIMIT {int(n)}"
    return con.execute(q).df()

def local_sql_recurring_merchants(con, min_count=3):
    q = f"""
    SELECT Description, COUNT(*) AS times, SUM(Amount) AS total_spend
    FROM transactions
    WHERE Is_Spend = TRUE
    GROUP BY Description
    HAVING COUNT(*) >= {int(min_count)}
    ORDER BY times DESC, total_spend DESC
    LIMIT 50
    """
    return con.execute(q).df()

def local_compute_kpis(con):
    row = con.execute("""
    SELECT
      COALESCE((SELECT SUM(Amount) FROM transactions WHERE Is_Spend = TRUE),0) AS total_spend,
      COALESCE((SELECT SUM(Credit_Amount) FROM transactions WHERE Is_Income = TRUE),0) AS total_income,
      COALESCE((SELECT COUNT(DISTINCT YearMonth) FROM transactions),0) AS months,
      COALESCE((SELECT COUNT(DISTINCT Description) FROM transactions),0) AS merchants
    """).df().iloc[0].to_dict()
    return row

@st.cache_data
def load_transactions(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    return df

def ensure_session_state():
    defaults = {
        'onboard_done': False,
        'user_name': '',
        'user_goal': '',
        'messages': [],
        'always_show_charts': False,
        'guided_advice': [],
        'chat_input': "",
        'last_chart': None,
        'onboard_msgs_appended': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except:
            pass
    st.rerun()

def llm_clarify_query(history: list, current_query: str) -> str:
    if not LLM_AVAILABLE:
        return current_query
    history_snip = history[-6:]
    history_lines = [f"[{m['role'].upper()}]: {m['content']}" for m in history_snip]
    system_prompt = (
        "You are a query clarifier. Rewrite the FINAL USER QUERY into a single explicit, self-contained "
        "financial question using HISTORY context. Output only the clarified question (no commentary)."
    )
    user_prompt = "HISTORY:\n" + "\n".join(history_lines) + f"\n\nFINAL USER QUERY: {current_query}\n\nCLARIFIED QUESTION:"
    messages = [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]
    try:
        clarified = simple_chat(messages, model="gpt-3.5-turbo", max_tokens=150, temperature=0.0)
        if clarified and isinstance(clarified, str) and clarified.strip():
            return clarified.strip()
        return current_query
    except:
        return current_query

def assistant_rewrite_with_llm(original_resp: str, sql_debug: Dict[str,str], top_facts: Optional[str], user_goal: str) -> str:
    if not LLM_AVAILABLE:
        return original_resp
    system = {
        "role":"system",
        "content": (
            "You are a concise factual assistant. Use only the numbers present in FACTS. "
            "Rewrite the input into a short polished message (1-3 sentences). Do NOT ask a follow-up question."
        )
    }
    user_lines = [
        f"USER GOAL: {user_goal or '(none)'}",
        "",
        "FACTS:",
        original_resp,
        "",
        "SQL:",
        sql_debug.get('sql','(none)')
    ]
    if top_facts:
        user_lines += ["", "TOP FACTS:", top_facts]
    user_lines += ["", "Rewrite into a concise polished message (no questions)."]
    messages = [system, {"role":"user","content":"\n".join(user_lines)}]
    try:
        polished = simple_chat(messages, model="gpt-3.5-turbo", max_tokens=220, temperature=0.2)
        if polished and isinstance(polished, str) and polished.strip():
            out = polished.strip()
            if out.endswith("?"):
                out = out.rstrip("?").strip(". ") + "."
            return out
        return original_resp
    except:
        return original_resp

def is_financial_query(query: str) -> bool:
    q = (query or "").lower()
    finance_keywords = [
        "spend", "spent", "income", "salary", "balance", "category", "categories",
        "merchant", "merchants", "refund", "refunds", "credit", "credits",
        "subscription", "subscriptions", "recurring", "savings", "save", "goal",
        "top", "last", "months", "years", "month", "year", "plot", "chart", "how much", "total"
    ]
    return any(k in q for k in finance_keywords)

def html_escape(text: str) -> str:
    """Basic HTML escaping for safe display in HTML blocks (not for JSON)."""
    return (text or "") \
        .replace("\\", "\\\\") \
        .replace("'", "\'") \
        .replace('"', '\\"') \
        .replace("&", "&amp;") \
        .replace("<", "&lt;") \
        .replace(">", "&gt;") \
        .replace("\n", "\\n")
        
def _convert_small_markdown_to_html(text: str) -> str:
    if not text:
        return ''
    esc = (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
    esc = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', esc)
    esc = re.sub(r'\*(.+?)\*', r'<em>\1</em>', esc)
    esc = esc.replace('\n', '<br/>')
    return esc

# --- Updated Chat Renderer (Fixes Chat Window Size/Ratio and Text) ---
def clear_and_render_chat():
    msgs = st.session_state.get('messages', [])

    html_lines = ['<div class="chat-history-container" role="log" aria-live="polite">']
    chart_objects = [] 

    for i, msg in enumerate(msgs):
        role = msg.get('role', 'assistant')
        content = msg.get('content', '')
        meta = msg.get('meta', {}) or {}
        bubble_class = "assistant" if role != "user" else "user"
        html_lines.append(f'<div class="chat-bubble {bubble_class}">')

        esc_content = _convert_small_markdown_to_html(content)
        html_lines.append(esc_content)

        fig_obj = meta.get('plotly_json_obj')
        if fig_obj and role != "user":
            chart_id = f"plotly-chart-{i}"
            html_lines.append('<hr>')
            html_lines.append(f'<div id="{chart_id}" style="height:320px; width:100%; margin-top:10px;"></div>')
            chart_objects.append({'id': chart_id, 'fig': fig_obj})

        sql_debug = meta.get('sql_debug') if isinstance(meta, dict) else None
        if sql_debug and isinstance(sql_debug, dict) and sql_debug.get('sql'):
            sql_esc = html_escape(sql_debug.get('sql', ''))
            reason = html_escape(sql_debug.get('reason', ''))
            html_lines.append('<div class="chat-meta">')
            html_lines.append(f'<details><summary>Show SQL & Rationale</summary>')
            html_lines.append(f'<pre>{sql_esc}</pre>')
            html_lines.append(f'<div style="margin-top:6px;color:rgba(0,0,0,0.6)">{reason}</div>')
            html_lines.append('</details>')
            html_lines.append('</div>')

        html_lines.append('</div>')
    html_lines.append('</div>')
    chat_html = "\n".join(html_lines)

    plotly_cdn = "https://cdn.plot.ly/plotly-2.22.0.min.js"
    chart_objects_json = json.dumps(chart_objects)

    # --- FIX IMPLEMENTED HERE: Use a fixed height for the Streamlit iframe ---
    # This ensures the chat component boundary remains constant, relying on the 
    # inner CSS (chat-history-container overflow: auto) for scrolling content.
    FIXED_CHAT_HEIGHT = 600 
    iframe_height = FIXED_CHAT_HEIGHT

    chat_iframe_css = get_chat_iframe_css()
    
    full_html = f"""
    {chat_iframe_css}

    <div>
      {chat_html}
    </div>

    <script src="{plotly_cdn}"></script>

    <script>
      (function() {{
        try {{
          const chartData = {chart_objects_json};
          chartData.forEach(item => {{
            const el = document.getElementById(item.id);
            if (!el) return;
            try {{
              const fig = item.fig;
              const config = {{responsive:true, displaylogo: false, modeBarButtonsToRemove: ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetview', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines', 'sendDataToCloud']}};
              Plotly.newPlot(el, fig.data, fig.layout || {{}}, config);
            }} catch (err) {{
              console.error("Failed to render Plotly chart for", item.id, err);
              el.innerHTML = '<div style="padding:12px;color:#666">Chart render failed (see console)</div>';
            }}
          }});
          const container = document.querySelector('.chat-history-container');
          // Auto-scroll to the bottom of the chat container
          if (container) container.scrollTop = container.scrollHeight;
        }} catch (e) {{
          console.error("chat render error", e);
        }}
      }})();
    </script>
    """

    components.html(full_html, height=iframe_height, scrolling=True)

# Note: The `get_chat_iframe_css()` definition from the previous response 
# must also be present in the final script.
# --- Main application (Logic and Wording Refined for Aesthetics) ---

def main_app():
    ensure_session_state()
    
    st.title("💸 Spendly – Chat-First Financial Assistant")     
    st.markdown("---")
    
    colL, colR = st.columns([2,1])
    with colL:
        # File uploader with clearer title/help
        uploaded = st.file_uploader("📂 Upload Transactions (Excel)", type=["xlsx","xls"], help="Upload your bank transaction Excel file.")
    with colR:
        # Checkbox refined
        use_default = st.checkbox("Use Sample Dataset", value=(not uploaded and os.path.exists(DEFAULT_XLSX)), help="Load the default anonymized bank data for demonstration.")
    
    df_raw = None
    if uploaded is not None:
        try:
            df_raw = load_transactions(uploaded, DEFAULT_SHEET)
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {e}")
            st.stop()
    elif use_default and os.path.exists(DEFAULT_XLSX):
        try:
            df_raw = load_transactions(DEFAULT_XLSX, DEFAULT_SHEET)
        except Exception as e:
            st.error(f"❌ Error reading default dataset: {e}")
            st.stop()
    else:
        st.info("⬆️ **Start here:** Please upload a file or select 'Use Sample Dataset' to load your data.")
        st.stop()
    
    with st.spinner("Preparing your data..."):
        # Preprocessing and DB setup (logic untouched)
        try:
            if RESPONDER_AVAILABLE and hasattr(responder, "preprocess"):
                df = responder.preprocess(df_raw)
            else:
                df = local_preprocess(df_raw)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.exception(e)
            st.stop()
        try:
            if RESPONDER_AVAILABLE and hasattr(responder, "init_db_from_df"):
                con = responder.init_db_from_df(df)
            else:
                con = local_init_db_from_df(df)
        except Exception as e:
            st.error(f"DB init failed: {e}")
            st.exception(e)
            st.stop()
            
    # KPI Display 
    try:
        if RESPONDER_AVAILABLE and hasattr(responder, "compute_kpis"):
            kpis = responder.compute_kpis(con)
        else:
            kpis = local_compute_kpis(con)
    except Exception:
        kpis = {"total_spend":0, "total_income":0, "months":0, "merchants":0}

    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Using '£' for currency. Changed wording and added emojis.
    col1.metric("Total Spend", f"£{float(kpis.get('total_spend',0)):,.0f}", help="Total amount debited across all transactions.") 
    col2.metric("Total Income", f"£{float(kpis.get('total_income',0)):,.0f}", help="Total amount credited across all transactions.")
    col3.metric("Months Covered", f"{int(kpis.get('months',0))}", help="Number of distinct months covered by the data.")
    col4.metric("Unique Payees", f"{int(kpis.get('merchants',0))}", help="Number of unique transaction descriptions (merchants).")
    
    # Quick Insight
    try:
        top_cat = responder.sql_top_categories(con, n=1) if RESPONDER_AVAILABLE else local_sql_top_categories(con, n=1)
        if not top_cat.empty:
            top_cat_label = top_cat['Category'].iloc[0]
            top_cat_amt = top_cat['spend'].iloc[0]
        else:
            top_cat_label = "N/A"
            top_cat_amt = 0
    except Exception:
        top_cat_label = "N/A"
        top_cat_amt = 0
    
    st.markdown(f"**Quick Insight:** Your largest spending category is **{top_cat_label}** (totaling **£{top_cat_amt:,.0f}**).")
    
    # Data Overview Expander (Aesthetic titles and charts)
    with st.expander("📊 Data Overview & Trends", expanded=True): 
        st.subheader("Top Categories (Spend)")
        try:
            full_top = responder.sql_top_categories(con, n=10) if RESPONDER_AVAILABLE else local_sql_top_categories(con, n=10)
            if not full_top.empty:
                st.dataframe(full_top, use_container_width=True)
            else:
                st.info("No category data available.")
        except Exception as e:
            st.error(f"Could not load categories: {e}")
            
        st.subheader("Monthly Spend Trend")
        try:
            monthly = con.execute("SELECT YearMonth, SUM(Amount) AS spend FROM transactions WHERE Is_Spend = TRUE GROUP BY YearMonth ORDER BY YearMonth").df()
            if not monthly.empty:
                # Aesthetic: Use the secondary color for the bar chart
                fig_all = go.Figure(go.Bar(x=monthly['YearMonth'], y=monthly['spend'], marker_color=SECONDARY_COLOR)) 
                fig_all.update_layout(
                    xaxis_title="Month", 
                    yaxis_title="Amount (£)", 
                    height=320,
                    template="plotly_white", 
                    margin=dict(t=10, l=40, r=20, b=40) 
                )
                st.plotly_chart(fig_all, use_container_width=True)
            else:
                st.info("No monthly spend data found.")
        except Exception as e:
            st.error(f"Could not render monthly trend: {e}")
            
    # Onboarding Flow (Refined wording and success message)
    if not st.session_state.onboard_done:
        with st.container():
            st.markdown("### 🎯 Define Your Goal") 
            with st.form("onboard_form", clear_on_submit=False):
                st.markdown("To get the best tailored advice, let's set a focus.")
                st.session_state.user_name = st.text_input("1. Your Name", value=st.session_state.user_name)
                st.session_state.user_goal = st.text_input("2. Financial Goal (e.g., 'reduce Amazon spend' or 'save for a trip')", value=st.session_state.user_goal, help="This helps the AI prioritize suggestions.")
                if st.form_submit_button("Start Chatting", type="primary"): 
                    st.session_state.onboard_done = True
                    st.success(f"✅ Setup complete! Your focus is: **{st.session_state.user_goal or 'No goal set'}**")
                    if not st.session_state.onboard_msgs_appended:
                        st.session_state.onboard_msgs_appended = True
                        base_suggestions = []
                        try:
                            if RESPONDER_AVAILABLE and hasattr(responder, "generate_guided_advice"):
                                base_suggestions = responder.generate_guided_advice(con, st.session_state.user_goal or "", top_n=3, use_llm=False)
                        except Exception:
                            base_suggestions = []
                        
                        initial_msg_content = f"Welcome **{st.session_state.user_name or 'User'}**! I'm Spendly, your financial AI. I'm ready to analyze your data.\n\n"
                        if base_suggestions:
                            initial_msg_content += "Here are some initial insights based on your spending patterns:"
                            st.session_state.messages.append({'role':'assistant','content': initial_msg_content})
                            for s in base_suggestions:
                                original = f"{s.get('title','Suggestion')}: {s.get('advice')}"
                                if s.get('impact_estimate') is not None:
                                    original += f" (est impact £{s.get('impact_estimate',0.0):,.2f})"
                                fact_sql = s.get('fact_sql','')
                                polished = assistant_rewrite_with_llm(original, {"sql":fact_sql,"reason":"suggestion"}, None, st.session_state.user_goal) if LLM_AVAILABLE else original
                                st.session_state.messages.append({'role':'assistant','content':polished,'meta':{'sql_debug':{"sql":"","reason":""}}})
                        else:
                            initial_msg_content += "What would you like to see first? Try asking 'What are my top 5 spend categories?'"
                            st.session_state.messages.append({'role':'assistant','content': initial_msg_content})

                        safe_rerun()
    else:
        if not st.session_state.messages:
            st.session_state.messages.insert(0, {'role':'assistant','content':f"Welcome back, **{st.session_state.user_name or 'User'}**! Ask me anything about your transactions."})


    # Render chat history and charts
    clear_and_render_chat()

    st.subheader('Ask a Question ✨')
    # Moved the 'always show charts' checkbox closer to the input
    st.session_state.always_show_charts = st.checkbox("Automatically generate charts when possible", value=st.session_state.always_show_charts, help="If unchecked, charts only display if you explicitly ask for one (e.g., 'show a chart').")
    
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([8,1,1])
        with cols[0]:
            user_q = st.text_input("Type your question or reply...", key="user_q_input", label_visibility="collapsed", placeholder="E.g., 'What are my top 5 categories this year?'")
        with cols[1]:
            send_pressed = st.form_submit_button("Send", type="primary")
        with cols[2]:
            show_chart_query = st.checkbox("Plot Result", value=False, key="chart_toggle_form", help="Check this box to specifically request a chart for this question.") 
            
    if send_pressed and user_q and user_q.strip():
        user_q = user_q.strip()
        st.session_state.messages.append({'role':'user','content': user_q})
        
        with st.status("Thinking... 🤔", expanded=False, state="running") as status: 
            status.update(label="Clarifying query... 💡", state="running") 
            clarified_q = llm_clarify_query(st.session_state.messages, user_q)
            
            if not is_financial_query(clarified_q):
                assistant_msg = "I can only answer questions about your transaction data. Please ask a financial query."
                st.session_state.messages.append({'role':'assistant','content': assistant_msg})
                status.update(label="Response Ready! ✨", state="complete", expanded=False)
                safe_rerun()
                return
                
            status.update(label="Processing intent and running SQL... 💾", state="running") 
            parsed = None
            explanation = None
            if LLM_AVAILABLE:
                try:
                    parsed, explanation = llm_parse_query_xml(clarified_q, known_categories=df['Category'].unique().tolist())
                except Exception:
                    parsed = None
            if parsed is None:
                parsed = {}
            parsed = dict(parsed)
            
            if show_chart_query or st.session_state.always_show_charts or "chart" in user_q.lower() or "plot" in user_q.lower():
                parsed['plot'] = True
                
            fig = None
            sql_debug = {"sql":"","reason":"responder missing"}
            resp_text = "Sorry — an internal error occurred."
            try:
                if RESPONDER_AVAILABLE:
                    resp_text, fig, sql_debug = responder.respond_to_parsed_intent(
                        con, parsed, clarified_q, known_categories=df['Category'].unique().tolist(),
                        show_charts_default=False, explanation=explanation
                    )
                else:
                    resp_text = "Sorry — the `responder` module is not available to process this query."
            except Exception as e:
                st.error(f"Error during query processing: {e}")
                traceback_str = traceback.format_exc()
                sql_debug['reason'] = f"Error: {str(e)} | Trace: {traceback_str[:300]}..."

            status.update(label="Polishing response... 🤖", state="running")
            
            # LLM rewrite for a polished answer (only if LLM is available)
            final_resp = assistant_rewrite_with_llm(resp_text, sql_debug, None, st.session_state.user_goal)

            msg_to_append = {
                'role':'assistant',
                'content': final_resp,
                'meta': {'sql_debug': sql_debug}
            }
            if fig is not None:
                # Convert Plotly figure to JSON serializable dict
                msg_to_append['meta']['plotly_json_obj'] = json.loads(pio.to_json(fig))

            st.session_state.messages.append(msg_to_append)
            
            status.update(label="Response Ready! ✨", state="complete", expanded=False)
            safe_rerun()
            
if __name__ == '__main__':
    main_app()