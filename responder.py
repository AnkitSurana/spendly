# responder.py
import re
import duckdb
import calendar
import pandas as pd
import plotly.graph_objs as go
from typing import Optional, Tuple, Dict, Any

# Optional LLM wrapper (non-fatal if missing)
try:
    from llm_client import simple_chat  # simple_chat(messages, max_tokens, temperature) -> text
except Exception:
    simple_chat = None

# --- SQL helpers (small set) ---
def init_db_from_df(df: pd.DataFrame, db_path=":memory:"):
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS transactions;")
    # Make sure necessary columns exist in df; allow missing via NULL AS
    expected_cols = [
        "Txn_Number", "Transaction_Date", "Txn_Type", "Description",
        "Debit_Amount", "Credit_Amount", "Amount", "Balance",
        "Category", "City", "Country", "Year", "Month", "YearMonth", "Week",
        "Is_Spend", "Is_Income"
    ]
    df_cols = {c.lower(): c for c in df.columns}
    select_parts = []
    alt_map = {}
    for col in expected_cols:
        lower = col.lower()
        if lower in df_cols:
            select_parts.append(f'"{df_cols[lower]}" AS "{col}"')
        else:
            # fall back to common alternatives if present otherwise NULL
            if col in df.columns:
                select_parts.append(f'"{col}" AS "{col}"')
            else:
                select_parts.append(f'NULL AS "{col}"')
    select_sql = ",\n    ".join(select_parts)
    create_sql = f"""
    CREATE TABLE transactions AS
    SELECT
      {select_sql}
    FROM df;
    """
    # register temp table
    con.register("df", df)
    con.execute(create_sql)
    con.unregister("df")
    try:
        con.execute("CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(Transaction_Date);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_category ON transactions(Category);")
    except Exception:
        pass
    return con

def compute_kpis(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    row = con.execute("""
        SELECT
          COALESCE((SELECT SUM(Amount) FROM transactions WHERE Is_Spend = TRUE),0) AS total_spend,
          COALESCE((SELECT SUM(Credit_Amount) FROM transactions WHERE Is_Income = TRUE),0) AS total_income,
          COALESCE((SELECT COUNT(DISTINCT YearMonth) FROM transactions),0) AS months,
          COALESCE((SELECT COUNT(DISTINCT Description) FROM transactions),0) AS merchants
    """).df().iloc[0].to_dict()
    return row

# Query builders
def sql_top_categories(con, n=10, months: Optional[int]=None):
    time_clause = ""
    if months:
        time_clause = f"AND Transaction_Date >= now() - INTERVAL {months} MONTH"
    q = f"""
        SELECT Category, (SUM(Debit_Amount) - SUM(Credit_Amount)) AS spend
        from transactions
        WHERE (Debit_Amount - Credit_Amount) > 0 {time_clause}
        GROUP BY 1
        ORDER BY spend DESC
        LIMIT {int(n)}
    """
    return con.execute(q).df()

def sql_spend_category_last_n_months(con, category: str, n: int):
    q = """
        WITH monthly AS (
            SELECT YearMonth, SUM(Amount) AS spend
            FROM transactions
            WHERE Is_Spend = TRUE
              AND lower(Category) LIKE '%' || lower(?) || '%'
            GROUP BY 1
            ORDER BY YearMonth DESC
            LIMIT ?
        )
        SELECT * FROM monthly ORDER BY YearMonth;
    """
    return con.execute(q, [category, int(n)]).df()

def sql_total_spend_last_n_months(con, n: int):
    q = """
        WITH monthly AS (
            SELECT YearMonth, SUM(Amount) AS spend
            FROM transactions
            WHERE Is_Spend = TRUE
            GROUP BY 1
            ORDER BY YearMonth DESC
            LIMIT ?
        )
        SELECT SUM(spend) AS total FROM monthly;
    """
    return con.execute(q, [int(n)]).df()

def sql_recurring_merchants(con, min_count=3):
    q = f"""
        SELECT Description, COUNT(*) AS times, SUM(Amount) AS total_spend
        FROM transactions
        WHERE Is_Spend = TRUE
        GROUP BY 1
        HAVING COUNT(*) >= {int(min_count)}
        ORDER BY times DESC, total_spend DESC
        LIMIT 50
    """
    return con.execute(q).df()

def sql_top_merchants(con, n=10, category: str=None, months: int=None):
    base = ["SELECT Description, SUM(Amount) AS spend, COUNT(*) AS times FROM transactions WHERE Is_Spend = TRUE"]
    params = []
    if category:
        base.append("AND lower(Category) LIKE '%' || lower(?) || '%'")
        params.append(category)
    if months:
        base.append("""
            AND YearMonth IN (
               SELECT YearMonth FROM (
                  SELECT DISTINCT YearMonth FROM transactions ORDER BY YearMonth DESC LIMIT ?
               )
            )
        """)
        params.append(int(months))
    base.append("GROUP BY 1 ORDER BY spend DESC LIMIT ?")
    params.append(int(n))
    q = "\n".join(base)
    return con.execute(q, params).df()

def respond_to_parsed_intent(
    con,
    parsed: Optional[dict],
    original_query: str,
    known_categories=None,
    show_charts_default=False,
    explanation: Optional[str] = None
) -> Tuple[str, Optional[go.Figure], Dict[str,str]]:
    """
    Extended to return (resp_text, fig, sql_debug)
    where sql_debug = {"sql": <sql string used>, "reason": <why we chose this query>}
    """

    # fallback regex parser if parsed is None
    def regex_fallback(q: str):
        ql = q.lower()
        if "top" in ql and "category" in ql:
            return {"intent":"TOP_CATEGORIES","category":None,"months":None,"plot":("show" in ql)}
        m = re.search(r"spend(?:ing)? on ([a-zA-Z &]+)", ql)
        if m:
            months = None
            mm = re.search(r"last (\d+) months", ql)
            if mm:
                months = int(mm.group(1))
            if "last month" in ql:
                months = 1
            return {"intent":"SPEND_ON_CATEGORY","category":m.group(1).strip(),"months":months,"plot":("show" in ql)}
        if "recurring" in ql or "subscription" in ql:
            return {"intent":"RECURRING","category":None,"months":None,"plot":("show" in ql)}
        if "income" in ql or "salary" in ql:
            return {"intent":"INCOME","category":None,"months":None,"plot":("show" in ql)}
        if "balance" in ql or "cash flow" in ql:
            return {"intent":"BALANCE","category":None,"months":None,"plot":("show" in ql)}
        mm = re.search(r"last (\d+) months", ql)
        if mm:
            return {"intent":"SPEND_TOTAL_PERIOD","category":None,"months":int(mm.group(1)),"plot":("show" in ql)}
        if "last month" in ql:
            return {"intent":"SPEND_TOTAL_PERIOD","category":None,"months":1,"plot":("show" in ql)}
        return {"intent":"UNKNOWN","category":None,"months":None,"plot":("show" in ql)}

    if parsed is None:
        parsed = regex_fallback(original_query)

    # Ensure parsed is mutable dict and normalise n as earlier patch
    parsed = dict(parsed)
    # recover n if missing
    m_n = re.search(r"\btop\s+(\d+)\b", original_query.lower())
    n_from_query = int(m_n.group(1)) if m_n else None
    if parsed.get("n") is None:
        if n_from_query:
            parsed["n"] = n_from_query
        else:
            if parsed.get("intent") == "TOP_MERCHANTS":
                parsed["n"] = 3
            elif parsed.get("intent") == "TOP_CATEGORIES":
                parsed["n"] = 5
            else:
                parsed["n"] = 10

    # Normalise category
    def best_category(name):
        if not name:
            return None
        nl = name.lower().strip()
        if known_categories:
            for c in known_categories:
                if c and c.lower() == nl:
                    return c
            for c in known_categories:
                if c and (nl in c.lower() or c.lower() in nl):
                    return c
        return name

    intent = parsed.get("intent", "UNKNOWN")
    category = best_category(parsed.get("category"))
    months = parsed.get("months")
    want_plot = parsed.get("plot") or show_charts_default

    resp_text = ""
    fig = None
    sql_debug = {"sql": "", "reason": ""}

    # --- map intents to SQL and reason text ---
    if intent == "TOP_CATEGORIES":
        n = parsed.get("n", 10)
        # SQL we will run (human-readable, parameter inserted)
        sql_debug["sql"] = f"""
SELECT Category, SUM(Amount) AS spend
FROM transactions
WHERE Is_Spend = TRUE
GROUP BY Category
ORDER BY spend DESC
LIMIT {n};
""".strip()
        sql_debug["reason"] = f"Top {n} spending categories across all time (filters Is_Spend to focus on outflows)."

        df_top = sql_top_categories(con, n=n, months=months)
        if not df_top.empty:
            resp_text = f"Top {n} categories: " + ", ".join([f"{row.Category} (£{row.spend:,.2f})" for _, row in df_top.iterrows()])
            if want_plot:
                fig = go.Figure(go.Bar(x=df_top['Category'], y=df_top['spend']))
                fig.update_layout(title="Top spending categories", xaxis_title="Category", yaxis_title="Amount (£)")
        else:
            resp_text = "No spending data found."

    elif intent == "SPEND_ON_CATEGORY":
        # category canonicalized earlier
        cat = category or parsed.get("category") or "requested category"

        # explicit month/year window from parsed (populated by LLM)
        month_num = parsed.get("month")           # 1..12 or None
        years_window = parsed.get("years")        # integer or None
        months_win = parsed.get("months")         # last-N-months (fallback)
        want_plot = parsed.get("plot") or show_charts_default

        # CASE A: user asked for a specific calendar month across last N years
        if month_num and years_window:
            # Build debug SQL (human-friendly)
            sql_debug["sql"] = f"""
SELECT Year, SUM(Amount) AS spend
FROM transactions
WHERE Is_Spend = TRUE
  AND lower(Category) LIKE '%{cat.lower()}%'
  AND Month = {int(month_num)}
  AND Year >= (EXTRACT(year FROM current_date) - {years_window - 1})
GROUP BY Year
ORDER BY Year;
""".strip()
            sql_debug["reason"] = (f"Per-year totals for category '{cat}' for month {int(month_num)} "
                                  f"over the last {int(years_window)} year(s).")

            # Run parameterised query: use years_window-1 as cutoff offset
            df_years = con.execute("""
                SELECT Year, SUM(Amount) AS spend
                FROM transactions
                WHERE Is_Spend = TRUE
                  AND lower(Category) LIKE '%' || lower(?) || '%'
                  AND Month = ?
                  AND Year >= (EXTRACT(year FROM current_date) - ?)
                GROUP BY Year
                ORDER BY Year;
            """, [cat, int(month_num), int(years_window - 1)]).df()

            total = float(df_years['spend'].sum()) if not df_years.empty else 0.0
            if not df_years.empty:
                per_lines = "; ".join([f"{int(r.Year)}: £{r.spend:,.2f}" for _, r in df_years.iterrows()])
                resp_text = f"Per-year spend for '{cat}' in {calendar.month_name[int(month_num)]}: {per_lines}. Total across {years_window} year(s): £{total:,.2f}."
                # optional chart
                if want_plot:
                    fig = go.Figure(go.Bar(x=df_years['Year'].astype(str), y=df_years['spend']))
                    fig.update_layout(title=f"{cat} spend — {calendar.month_name[int(month_num)]} (per year)", xaxis_title="Year", yaxis_title="Amount (£)")
            else:
                # build a safe month name (fallback to numeric if needed)
                try:
                    month_name = calendar.month_name[int(month_num)] if month_num and 1 <= int(month_num) <= 12 else f"month #{month_num}"
                except Exception:
                    month_name = f"month #{month_num}"
                
                resp_text = f"No spending found for '{cat}' in {month_name} over the last {years_window} year(s)."


        # CASE B: user requested recent months (last N months)
        else:
            n_months = months_win or 3
            sql_debug["sql"] = f"""
SELECT YearMonth, SUM(Amount) AS spend
FROM transactions
WHERE Is_Spend = TRUE
  AND lower(Category) LIKE '%{cat.lower()}%'
GROUP BY YearMonth
ORDER BY YearMonth
LIMIT {int(n_months)};
""".strip()
            sql_debug["reason"] = f"Monthly spend for category '{cat}' over the last {int(n_months)} month(s)."

            df_cat = sql_spend_category_last_n_months(con, category=cat, n=n_months)
            total = float(df_cat['spend'].sum()) if not df_cat.empty else 0.0
            resp_text = f"You spent £{total:,.2f} on '{cat}' in the last {n_months} month(s)."
            if want_plot and not df_cat.empty:
                fig = go.Figure(go.Bar(x=df_cat['YearMonth'], y=df_cat['spend']))
                fig.update_layout(title=f"Monthly spend — {cat}", xaxis_title="Month", yaxis_title="Amount (£)")

    elif intent == "SPEND_TOTAL_PERIOD":
        months_win = months or 1
        sql_debug["sql"] = f"""
WITH monthly AS (
  SELECT YearMonth, SUM(Amount) AS spend
  FROM transactions
  WHERE Is_Spend = TRUE
  GROUP BY YearMonth
  ORDER BY YearMonth DESC
  LIMIT {months_win}
)
SELECT SUM(spend) as total FROM monthly;
""".strip()
        sql_debug["reason"] = f"Summing monthly spends for the last {months_win} month(s) gives total outflow in the requested window."

        df_total = sql_total_spend_last_n_months(con, n=months_win)
        total = float(df_total.iloc[0]['total']) if not df_total.empty and pd.notna(df_total.iloc[0]['total']) else 0.0
        resp_text = f"Total spend in the last {months_win} month(s): £{total:,.2f}."
        if want_plot:
            breakdown = con.execute("""
                SELECT YearMonth, SUM(Amount) AS spend
                FROM transactions WHERE Is_Spend = TRUE
                GROUP BY YearMonth ORDER BY YearMonth DESC LIMIT ?
            """, [int(months_win)]).df().sort_values('YearMonth')
            if not breakdown.empty:
                fig = go.Figure(go.Bar(x=breakdown['YearMonth'], y=breakdown['spend']))
                fig.update_layout(title="Monthly spend (last N months)", xaxis_title="Month", yaxis_title="Amount (£)")

    elif intent == "TOP_MERCHANTS":
        n = int(parsed.get("n") or 10)
        want_plot = parsed.get("plot") or show_charts_default

        # detect if user asked about refunds/credits via parsed fields or raw text fallback
        raw_lower = original_query.lower() if original_query else ""
        credit_requested = False
        # check common words for refunds/credits
        if any(k in raw_lower for k in ["refund", "refunds", "credited", "credit", "reimburse", "reimbursement"]):
            credit_requested = True

        if credit_requested:
            # Use credit aggregation (refunds)
            sql_debug["sql"] = f"""
SELECT Description, SUM(Credit_Amount) AS total_refund, COUNT(*) AS times
FROM transactions
WHERE Credit_Amount > 0
  AND Year = (EXTRACT(year FROM current_date) - 1)
GROUP BY Description
ORDER BY total_refund DESC
LIMIT {n};
""".strip()
            sql_debug["reason"] = "User asked about refunds/credits; aggregating Credit_Amount to identify biggest refunds last year."

            # parameterize: last year
            dfm = con.execute("""
                SELECT Description, SUM(Credit_Amount) AS total_refund, COUNT(*) AS times
                FROM transactions
                WHERE Credit_Amount > 0
                  AND Year = (EXTRACT(year FROM current_date) - 1)
                GROUP BY Description
                ORDER BY total_refund DESC
                LIMIT ?
            """, [n]).df()

            if not dfm.empty:
                resp_text = f"Top {n} refunds last year: " + ", ".join([f"{r.Description} (refunds totalling £{r.total_refund:,.2f})" for _, r in dfm.iterrows()])
                if want_plot:
                    fig = go.Figure(go.Bar(x=dfm['Description'], y=dfm['total_refund']))
                    fig.update_layout(title="Top refunds last year", xaxis_title="Merchant", yaxis_title="Refund (£)")
            else:
                resp_text = "No refunds (credits) found for last year."

        else:
            # Regular spend-based top merchants
            sql_debug["sql"] = f"""
SELECT Description, SUM(Amount) AS spend, COUNT(*) AS times
FROM transactions
WHERE Is_Spend = TRUE
GROUP BY Description
ORDER BY spend DESC
LIMIT {n};
""".strip()
            sql_debug["reason"] = "Aggregate merchant spend to show top merchants by total spent."

            dfm = sql_top_merchants(con, n=n, category=category)
            if not dfm.empty:
                resp_text = (f"Top {n} merchants in {category}: " if category else f"Top {n} merchants overall: ") + ", ".join([f"{r.Description} (£{r.spend:,.2f})" for _, r in dfm.iterrows()]) + "."
                if want_plot:
                    fig = go.Figure(go.Bar(x=dfm['Description'], y=dfm['spend']))
                    fig.update_layout(title="Top merchants", xaxis_title="Merchant", yaxis_title="Amount (£)")
            else:
                resp_text = "No merchant spend found."

    elif intent == "RECURRING":
        sql_debug["sql"] = """
SELECT Description, COUNT(*) AS times, SUM(Amount) AS total_spend
FROM transactions
WHERE Is_Spend = TRUE
GROUP BY Description
HAVING COUNT(*) >= 3
ORDER BY times DESC, total_spend DESC
LIMIT 50;
""".strip()
        sql_debug["reason"] = "Find merchants with frequent transactions (>=3) to detect likely subscriptions or recurring charges."

        df_rec = sql_recurring_merchants(con)
        if not df_rec.empty:
            top = df_rec.head(10)
            resp_text = "Likely recurring merchants: " + ", ".join([f"{r.Description} (x{int(r.times)})" for _, r in top.iterrows()]) + "."
            if want_plot:
                fig = go.Figure(go.Bar(x=top['Description'], y=top['times']))
                fig.update_layout(title="Recurring merchants (count)", xaxis_title="Merchant", yaxis_title="Count")
        else:
            resp_text = "No recurring merchants detected."

    elif intent == "INCOME":
        sql_debug["sql"] = """
SELECT YearMonth, SUM(Credit_Amount) AS income
FROM transactions
WHERE Is_Income = TRUE
GROUP BY YearMonth
ORDER BY YearMonth;
""".strip()
        sql_debug["reason"] = "Aggregate credited amounts by month to show income trend."

        df_inc = sql_income_summary(con)
        total_inc = df_inc['income'].sum() if not df_inc.empty else 0.0
        resp_text = f"Total recorded income: £{total_inc:,.2f}."
        if want_plot and not df_inc.empty:
            fig = go.Figure(go.Scatter(x=df_inc['YearMonth'], y=df_inc['income'], mode='lines+markers'))
            fig.update_layout(title="Monthly income", xaxis_title="Month", yaxis_title="Income (£)")

    elif intent == "BALANCE":
        sql_debug["sql"] = """
SELECT YearMonth,
       AVG(Balance) AS avg_balance,
       STDDEV(Balance) AS std_balance,
       MIN(Balance) AS min_balance,
       MAX(Balance) AS max_balance
FROM transactions
WHERE Balance IS NOT NULL
GROUP BY YearMonth
ORDER BY YearMonth;
""".strip()
        sql_debug["reason"] = "Compute monthly balance statistics to show volatility and min/max range."

        df_bal = sql_balance_stats(con)
        if not df_bal.empty:
            current_avg = df_bal['avg_balance'].iloc[-1]
            resp_text = f"Balance volatility (std): {df_bal['std_balance'].mean():.2f}. Current avg balance: £{current_avg:,.2f}."
            if want_plot:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_bal['YearMonth'], y=df_bal['avg_balance'], mode='lines+markers', name='Avg Balance'))
                fig.add_trace(go.Scatter(x=df_bal['YearMonth'], y=df_bal['min_balance'], mode='lines', name='Min'))
                fig.add_trace(go.Scatter(x=df_bal['YearMonth'], y=df_bal['max_balance'], mode='lines', name='Max'))
                fig.update_layout(title="Balance stats", xaxis_title="Month", yaxis_title="£")
        else:
            resp_text = "No balance data available."

    else:
        resp_text = "Sorry, I couldn't classify your question. Try: 'how much did I spend last month', 'spending on groceries last 3 months', 'top 5 merchants', 'recurring'."
        sql_debug["sql"] = ""
        sql_debug["reason"] = "No recognized intent."

    # Prepend explanation (LLM-provided short plan) if provided
    if explanation:
        cleaned = explanation.strip()
        if cleaned:
            resp_text = cleaned + "\n\n" + resp_text

    return resp_text, fig, sql_debug

# ---------------------------
# New helpers: simulate_cut() and generate_guided_advice()
# ---------------------------

def simulate_cut(con: duckdb.DuckDBPyConnection, category: str, pct: float = 10.0, months_window: int = 3) -> Tuple[float, float]:
    """
    Estimate savings if the user cuts `pct` percent from `category`, using the last `months_window` months.
    Returns (avg_baseline, estimated_saving).
    """
    try:
        df = con.execute("""
            WITH recent AS (
              SELECT YearMonth, SUM(Amount) AS spend
              FROM transactions
              WHERE Is_Spend = TRUE
                AND lower(Category) LIKE '%' || lower(?) || '%'
              GROUP BY YearMonth
              ORDER BY YearMonth DESC
              LIMIT ?
            )
            SELECT AVG(spend) as avg_spend FROM recent;
        """, [category, int(months_window)]).df()
        avg = float(df['avg_spend'].iloc[0]) if not df.empty and pd.notna(df.iloc[0]['avg_spend']) else 0.0
        saving = avg * (pct / 100.0)
        return round(avg, 2), round(saving, 2)
    except Exception:
        return 0.0, 0.0

def generate_guided_advice(con: duckdb.DuckDBPyConnection, user_goal: str, top_n: int = 3, use_llm: bool = True):
    """
    Produce up to `top_n` data-grounded suggestions tailored to user_goal.
    Each suggestion is a dict: {
      title, advice (short str), impact_estimate (GBP float), fact_sql (str), fact_sample (pd.DataFrame)
    }
    The numbers are computed locally; LLM (if available) is used only to rewrite text and must not invent numbers.
    """
    suggestions = []
    # Gather facts
    try:
        df_top_cat = sql_top_categories(con, n=5)
    except Exception:
        df_top_cat = pd.DataFrame()
    try:
        df_rec = sql_recurring_merchants(con, min_count=3)
    except Exception:
        df_rec = pd.DataFrame()
    try:
        df_merch = sql_top_merchants(con, n=10)
    except Exception:
        df_merch = pd.DataFrame()
    kpis = compute_kpis(con)

    # Candidate generation (data-first)
    candidates = []

    # merchant hint: if user mentions a merchant name, prioritize that candidate
    merchant_hint = None
    goal_low = (user_goal or "").lower()
    if not df_merch.empty:
        for desc in df_merch['Description'].astype(str).unique():
            if desc and desc.lower() in goal_low:
                merchant_hint = desc
                break
    if merchant_hint:
        tot = float(con.execute("SELECT COALESCE(SUM(Amount),0) FROM transactions WHERE lower(Description) LIKE '%' || lower(?) || '%'", [merchant_hint]).fetchone()[0] or 0)
        sample = con.execute("SELECT Transaction_Date, Description, Amount FROM transactions WHERE lower(Description) LIKE '%' || lower(?) || '%' ORDER BY Transaction_Date DESC LIMIT 5", [merchant_hint]).df()
        candidates.append({
            "title": f"Cap spend at {merchant_hint}",
            "advice": f"You spent ~£{tot:,.2f} at {merchant_hint}. Consider limiting frequency or setting a monthly cap.",
            "impact_estimate": round(tot * 0.10, 2),
            "fact_sql": f"SELECT SUM(Amount) FROM transactions WHERE lower(Description) LIKE '%{merchant_hint.lower()}%';",
            "fact_sample": sample
        })

    # top category trim
    if not df_top_cat.empty:
        top_cat = df_top_cat.iloc[0]
        cat_name = top_cat['Category']
        cat_sum = float(top_cat['spend'])
        sample = con.execute("SELECT Transaction_Date, Description, Amount FROM transactions WHERE Is_Spend = TRUE AND lower(Category) LIKE '%' || lower(?) || '%' ORDER BY Transaction_Date DESC LIMIT 5", [cat_name]).df()
        candidates.append({
            "title": f"Trim {cat_name} by 10%",
            "advice": f"Your {cat_name} spend totals £{cat_sum:,.2f}. A 10% cut could free ≈ £{cat_sum*0.10:,.2f}.",
            "impact_estimate": round(cat_sum * 0.10, 2),
            "fact_sql": f"SELECT SUM(Amount) FROM transactions WHERE lower(Category) LIKE '%{cat_name.lower()}%';",
            "fact_sample": sample
        })

    # recurring subscriptions
    if not df_rec.empty:
        sample_rec = df_rec.head(3)
        total_rec_sample = float(sample_rec['total_spend'][:3].sum()) if len(sample_rec) > 0 else 0.0
        candidates.append({
            "title": "Review recurring subscriptions",
            "advice": f"Detected recurring charges: {', '.join(sample_rec['Description'].head(3).tolist())}. Cancelling unused subs frees monthly cash.",
            "impact_estimate": round(total_rec_sample * 0.5, 2),
            "fact_sql": "SELECT Description, COUNT(*) AS times, SUM(Amount) AS total_spend FROM transactions WHERE Is_Spend = TRUE GROUP BY Description HAVING COUNT(*)>=3 ORDER BY times DESC LIMIT 10;",
            "fact_sample": sample_rec[['Description','times','total_spend']] if not sample_rec.empty else pd.DataFrame()
        })

    # automation vs buffer
    months = int(kpis.get('months') or 1)
    avg_monthly_spend = (float(kpis.get('total_spend') or 0) / months) if months>0 else 0.0
    avg_monthly_income = (float(kpis.get('total_income') or 0) / months) if months>0 else 0.0
    if avg_monthly_income > avg_monthly_spend and (avg_monthly_income - avg_monthly_spend) > 0:
        candidates.append({
            "title": "Automate 10% savings",
            "advice": f"Automate 10% of average income (£{avg_monthly_income:,.2f}) each month to build savings effortlessly.",
            "impact_estimate": round(avg_monthly_income * 0.10, 2),
            "fact_sql": "(derived from total income / months)",
            "fact_sample": pd.DataFrame([{"avg_income": avg_monthly_income, "avg_spend": avg_monthly_spend}])
        })
    else:
        candidates.append({
            "title": "Build 1-month emergency buffer",
            "advice": f"Target a buffer equal to one month's average spend (£{avg_monthly_spend:,.2f}).",
            "impact_estimate": round(avg_monthly_spend, 2),
            "fact_sql": "(derived from total spend / months)",
            "fact_sample": pd.DataFrame([{"avg_spend": avg_monthly_spend}])
        })

    # choose top candidates by estimated impact
    candidates = sorted(candidates, key=lambda x: x.get('impact_estimate', 0), reverse=True)[:top_n]

    # optionally use LLM to rewrite/shorten advice (LLM must not invent numbers)
    final_suggestions = []
    if use_llm and simple_chat:
        # create a constrained prompt (facts + candidates)
        facts_lines = []
        for c in candidates:
            facts_lines.append(f"- {c['title']}: est=£{c['impact_estimate']:,.2f}; sql={c['fact_sql']}")
        prompt = "User goal: {}\nFacts:\n{}\n\nProduce a JSON array of objects with {title, explanation<=25 words} using only the numbers above. Keep concise.".format(user_goal, "\n".join(facts_lines))
        try:
            raw = simple_chat([{"role":"system","content":"You are a concise assistant."},{"role":"user","content":prompt}], max_tokens=300, temperature=0.2)
            import json
            m = re.search(r"(\[.*\])", raw, re.DOTALL)
            json_text = m.group(1) if m else raw
            parsed = json.loads(json_text)
            for i, p in enumerate(parsed):
                base = candidates[i] if i < len(candidates) else candidates[0]
                final_suggestions.append({
                    "title": p.get("title", base["title"]),
                    "advice": p.get("explanation", base.get("advice", "")),
                    "impact_estimate": base.get("impact_estimate", 0.0),
                    "fact_sql": base.get("fact_sql", ""),
                    "fact_sample": base.get("fact_sample", pd.DataFrame())
                })
        except Exception:
            # LLM failed -> fallback deterministic
            for c in candidates:
                final_suggestions.append({
                    "title": c['title'],
                    "advice": c['advice'],
                    "impact_estimate": c['impact_estimate'],
                    "fact_sql": c['fact_sql'],
                    "fact_sample": c['fact_sample']
                })
    else:
        # deterministic: just return the candidates
        for c in candidates:
            final_suggestions.append({
                "title": c['title'],
                "advice": c['advice'],
                "impact_estimate": c['impact_estimate'],
                "fact_sql": c['fact_sql'],
                "fact_sample": c['fact_sample']
            })

    return final_suggestions
