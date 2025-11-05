# # prompts.py
# # central place for LLM prompt templates (editable)

# PARSER_XML_SYSTEM = "You are a strict XML parser for short banking queries. Return ONLY the compact <result>...XML block (no commentary)."

# PARSER_XML_USER = """
# Parse the user query into a single <result> XML block with fields:
# <intent> one of: TOP_CATEGORIES, SPEND_ON_CATEGORY, SPEND_TOTAL_PERIOD, TOP_MERCHANTS, RECURRING, INCOME, BALANCE, UNKNOWN
# <category> category name or empty
# <months> integer or empty
# <plot> true or false
# <explain> short explanation <= 40 words

# Known categories: {known_categories}

# User query:
# {query}

# Output example (single line, exactly one XML block):
# <result><intent>SPEND_ON_CATEGORY</intent><category>groceries</category><months>3</months><plot>false</plot><explain>I'll fetch monthly spend for groceries over the last 3 months.</explain></result>
# """
# # For optional LLM coaching/advice
# ADVICE_SYSTEM = "You are a friendly financial coach. Provide practical, low-risk suggestions."
# ADVICE_USER = "Summary: {summary}\nProvide: 3 practical actions, 1 quick sanity check, and a two-sentence motivational closing. Keep brief and non-prescriptive."

# prompts.py
# prompts.py
"""
Canonical prompt templates and many-shot examples for intent->XML parsing.
This file now contains a large set of diverse examples to help the LLM canonicalise user queries.
"""

CANONICAL_SCHEMA_DESC = (
    "Output MUST be a single-line XML block <result>...</result>. Fields (if not applicable, use empty tag):\n"
    " - <intent>: one of [TOP_CATEGORIES, SPEND_ON_CATEGORY, SPEND_TOTAL_PERIOD, TOP_MERCHANTS, RECURRING, INCOME, BALANCE, UNKNOWN]\n"
    " - <category>: normalized category string or empty\n"
    " - <category_confidence>: one of [high, medium, low] indicating certainty about category mapping\n"
    " - <n>: integer for top-N requests (or empty)\n"
    " - <months>: integer for last-N-months (or empty)\n"
    " - <month>: integer 1..12 for a specific calendar month (or empty)\n"
    " - <years>: integer for last-N-years (or empty)\n"
    " - <plot>: true/false\n"
    " - <explain>: one short sentence (<= 30 words) describing why this mapping was chosen\n"
)

# Expanded example bank: 48 examples covering many phrasings and edge cases.
EXAMPLES = [
    # 1
    {"query": "top 3 merchants spent on in the last 3 months",
     "xml": "<result><intent>TOP_MERCHANTS</intent><category></category><category_confidence>low</category_confidence><n>3</n><months>3</months><month></month><years></years><plot>false</plot><explain>Return top 3 merchants by spend for the last 3 months.</explain></result>"},
    # 2
    {"query": "what was my total spend in groceries in april month for last 2 years",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months></months><month>4</month><years>2</years><plot>false</plot><explain>Per-year April totals for Groceries across the last 2 years and a combined total.</explain></result>"},
    # 3
    {"query": "how much did i spend last month",
     "xml": "<result><intent>SPEND_TOTAL_PERIOD</intent><category></category><category_confidence>low</category_confidence><n></n><months>1</months><month></month><years></years><plot>false</plot><explain>Sum of all spending in the most recent calendar month.</explain></result>"},
    # 4
    {"query": "show recurring subscriptions",
     "xml": "<result><intent>RECURRING</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Detect merchants with repeated charges, likely subscriptions.</explain></result>"},
    # 5
    {"query": "spending on groceries last 3 months show graph",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months>3</months><month></month><years></years><plot>true</plot><explain>Monthly trend for Groceries for last 3 months and chart requested.</explain></result>"},
    # 6
    {"query": "what are my top spending categories",
     "xml": "<result><intent>TOP_CATEGORIES</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Return highest spending categories overall; app will apply default top-N.</explain></result>"},
    # 7
    {"query": "top 5 categories last 6 months",
     "xml": "<result><intent>TOP_CATEGORIES</intent><category></category><category_confidence>low</category_confidence><n>5</n><months>6</months><month></month><years></years><plot>false</plot><explain>Top five categories measured over the last six months.</explain></result>"},
    # 8
    {"query": "top 10 merchants for groceries this year",
     "xml": "<result><intent>TOP_MERCHANTS</intent><category>Groceries</category><category_confidence>high</category_confidence><n>10</n><months></months><month></month><years>1</years><plot>false</plot><explain>Top 10 merchants within Groceries for the current year.</explain></result>"},
    # 9
    {"query": "plot my income for the last 12 months",
     "xml": "<result><intent>INCOME</intent><category></category><category_confidence>low</category_confidence><n></n><months>12</months><month></month><years></years><plot>true</plot><explain>Monthly income totals for the past 12 months as a time series.</explain></result>"},
    # 10
    {"query": "show balance volatility and min max by month",
     "xml": "<result><intent>BALANCE</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>true</plot><explain>Compute monthly balance stats to show volatility and range.</explain></result>"},
    # 11
    {"query": "how much on rent",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Rent</category><category_confidence>medium</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Assume 'rent' is a category; return spend on Rent (ask for timeframe if needed).</explain></result>"},
    # 12
    {"query": "spending at coffee shop last month, show chart",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Coffee</category><category_confidence>low</category_confidence><n></n><months>1</months><month></month><years></years><plot>true</plot><explain>Monthly spend for Coffee merchants in the last month with chart.</explain></result>"},
    # 13
    {"query": "give me spend for April for the past 2 years on groceries",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months></months><month>4</month><years>2</years><plot>false</plot><explain>Per-year April spend for Groceries over the last two years, plus total.</explain></result>"},
    # 14
    {"query": "what were my spends in the last two Aprils for groceries",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months></months><month>4</month><years>2</years><plot>false</plot><explain>Interpret as Aprils from the last two years and return per-year totals.</explain></result>"},
    # 15
    {"query": "top merchants overall",
     "xml": "<result><intent>TOP_MERCHANTS</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Return top merchants overall; default window applied by the app.</explain></result>"},
    # 16
    {"query": "give me raw transactions since jan",
     "xml": "<result><intent>UNKNOWN</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>User requests raw export; system should confirm and offer CSV download.</explain></result>"},
    # 17
    {"query": "last 6 months spend groceries",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months>6</months><month></month><years></years><plot>false</plot><explain>Sum of monthly spends for Groceries across the last 6 months.</explain></result>"},
    # 18
    {"query": "do I have any subscriptions showing recurring charges?",
     "xml": "<result><intent>RECURRING</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Detect merchants with frequent repeat charges to highlight subscriptions.</explain></result>"},
    # 19
    {"query": "how much did I spend on Uber in March 2024",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Transport</category><category_confidence>medium</category_confidence><n></n><months></months><month>3</month><years>1</years><plot>false</plot><explain>Spend on transport merchants (Uber) for March 2024; year expressed as relative 'last 1 year'.</explain></result>"},
    # 20
    {"query": "show me last 24 months spending trend",
     "xml": "<result><intent>SPEND_TOTAL_PERIOD</intent><category></category><category_confidence>low</category_confidence><n></n><months>24</months><month></month><years></years><plot>true</plot><explain>Monthly total spend over the last 24 months as a trend chart.</explain></result>"},
    # 21
    {"query": "what did I spend on groceries in jan 2024 and jan 2023",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months></months><month>1</month><years>2</years><plot>false</plot><explain>Per-year January totals across last two Januaries for Groceries.</explain></result>"},
    # 22
    {"query": "top 7 categories this year",
     "xml": "<result><intent>TOP_CATEGORIES</intent><category></category><category_confidence>low</category_confidence><n>7</n><months></months><month></month><years>1</years><plot>false</plot><explain>Top seven spending categories for the current year by total spend.</explain></result>"},
    # 23
    {"query": "plot groceries vs dining out last 6 months",
     "xml": "<result><intent>TOP_CATEGORIES</intent><category></category><category_confidence>low</category_confidence><n></n><months>6</months><month></month><years></years><plot>true</plot><explain>Compare categories Groceries and Dining Out over last 6 months as a chart.</explain></result>"},
    # 24
    {"query": "how much salary have I received this year",
     "xml": "<result><intent>INCOME</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years>1</years><plot>false</plot><explain>Total credited income for the current year (year-to-date).</explain></result>"},
    # 25
    {"query": "give me min and max balance per month for last 12 months",
     "xml": "<result><intent>BALANCE</intent><category></category><category_confidence>low</category_confidence><n></n><months>12</months><month></month><years></years><plot>true</plot><explain>Monthly min/max/avg balance stats over past 12 months for volatility analysis.</explain></result>"},
    # 26
    {"query": "spent on tesco last month",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>medium</category_confidence><n></n><months>1</months><month></month><years></years><plot>false</plot><explain>Map 'Tesco' to Groceries category and calculate last month's spend.</explain></result>"},
    # 27
    {"query": "list subscriptions and amount per month",
     "xml": "<result><intent>RECURRING</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Identify merchants with regular monthly charges and estimate monthly amount.</explain></result>"},
    # 28
    {"query": "how much did i spend on groceries in march",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months></months><month>3</month><years></years><plot>false</plot><explain>Month-specific spend for Groceries for March (unspecified year implies recent years window if needed).</explain></result>"},
    # 29
    {"query": "top 3 categories in last 90 days",
     "xml": "<result><intent>TOP_CATEGORIES</intent><category></category><category_confidence>low</category_confidence><n>3</n><months>3</months><month></month><years></years><plot>false</plot><explain>Top 3 categories over an approximate 90-day window (last 3 months).</explain></result>"},
    # 30
    {"query": "did I spend more on rent or utilities this year",
     "xml": "<result><intent>TOP_CATEGORIES</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years>1</years><plot>false</plot><explain>Compare category totals Rent vs Utilities for the current year; present higher category.</explain></result>"},
    # 31
    {"query": "download transactions since 01-01-2024 as csv",
     "xml": "<result><intent>UNKNOWN</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>User requests raw export; system should ask to confirm and provide CSV download.</explain></result>"},
    # 32
    {"query": "how much did I spend on coffee last week",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Coffee</category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Recent period 'last week' mapped to short window; compute spend for Coffee merchants in that period.</explain></result>"},
    # 33
    {"query": "spending on groceries for March and April this year",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months></months><month></month><years>1</years><plot>false</plot><explain>Return monthly spends for March and April for Groceries this year, grouped by month.</explain></result>"},
    # 34
    {"query": "what's my average monthly grocery spend",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>medium</category_confidence><n></n><months>12</months><month></month><years></years><plot>false</plot><explain>Compute average monthly spend in Groceries over a default recent window (12 months).</explain></result>"},
    # 35
    {"query": "show a chart of my top merchants last month",
     "xml": "<result><intent>TOP_MERCHANTS</intent><category></category><category_confidence>low</category_confidence><n></n><months>1</months><month></month><years></years><plot>true</plot><explain>Visualise top merchants' spend for last month as a chart.</explain></result>"},
    # 36
    {"query": "did I get paid twice in July 2024",
     "xml": "<result><intent>INCOME</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month>7</month><years>1</years><plot>false</plot><explain>Inspect credited transactions for July 2024 to check income occurrences.</explain></result>"},
    # 37
    {"query": "compare monthly grocery spend between 2023 and 2024",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months></months><month></month><years>2</years><plot>true</plot><explain>Compare Groceries month-by-month for 2023 vs 2024 as a chart.</explain></result>"},
    # 38
    {"query": "list top merchants where I spent more than £1000",
     "xml": "<result><intent>TOP_MERCHANTS</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Filter merchant totals where spend exceeds threshold; return top merchants matching condition.</explain></result>"},
    # 39
    {"query": "how volatile is my balance over the last 6 months",
     "xml": "<result><intent>BALANCE</intent><category></category><category_confidence>low</category_confidence><n></n><months>6</months><month></month><years></years><plot>true</plot><explain>Compute volatility metrics for balance over last 6 months and show chart.</explain></result>"},
    # 40
    {"query": "spent 20 on amazon yesterday - how to correct",
     "xml": "<result><intent>UNKNOWN</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Ambiguous correction request; system should ask for clarification and transaction details.</explain></result>"},
    # 41
    {"query": "show me the biggest refunds last year",
     "xml": "<result><intent>TOP_MERCHANTS</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month></month><years>1</years><plot>false</plot><explain>Identify large positive credit amounts (refunds) and return top occurrences for last year.</explain></result>"},
    # 42
    {"query": "how much have i spent on groceries year to date",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Groceries</category><category_confidence>high</category_confidence><n></n><months></months><month></month><years>1</years><plot>false</plot><explain>Sum of Groceries spend from start of current year to date (year-to-date).</explain></result>"},
    # 43
    {"query": "top subscriptions this month",
     "xml": "<result><intent>RECURRING</intent><category></category><category_confidence>low</category_confidence><n></n><months>1</months><month></month><years></years><plot>false</plot><explain>Detect likely subscription merchants with recurring charges within this month.</explain></result>"},
    # 44
    {"query": "spend on petrol for last 6 months",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Transport</category><category_confidence>medium</category_confidence><n></n><months>6</months><month></month><years></years><plot>false</plot><explain>Aggregate petrol/transport-related spend across the last six months.</explain></result>"},
    # 45
    {"query": "I want a plot of my last 3 months expenses grouped by category",
     "xml": "<result><intent>TOP_CATEGORIES</intent><category></category><category_confidence>low</category_confidence><n></n><months>3</months><month></month><years></years><plot>true</plot><explain>Group spending by category for the last 3 months and show as chart.</explain></result>"},
    # 46
    {"query": "which merchants charged me most in Oct 2024",
     "xml": "<result><intent>TOP_MERCHANTS</intent><category></category><category_confidence>low</category_confidence><n></n><months></months><month>10</month><years>1</years><plot>false</plot><explain>List merchants with highest spend for October 2024 by summing transaction amounts.</explain></result>"},
    # 47
    {"query": "how much did I spend on fuel vs groceries last month",
     "xml": "<result><intent>TOP_CATEGORIES</intent><category></category><category_confidence>low</category_confidence><n></n><months>1</months><month></month><years></years><plot>true</plot><explain>Compare fuel vs groceries totals for last month; present comparative chart.</explain></result>"},
    # 48
    {"query": "did I receive any bank fees this quarter",
     "xml": "<result><intent>SPEND_ON_CATEGORY</intent><category>Bank Fees</category><category_confidence>medium</category_confidence><n></n><months></months><month></month><years></years><plot>false</plot><explain>Search for transactions mapped to Bank Fees during the current quarter and report totals.</explain></result>"},
]

SYSTEM_INSTRUCTION = (
    "You are a strict parser. Given a user's short question about bank transactions, "
    "produce exactly ONE single-line XML <result>...</result> as defined below. Do not add any extra text.\n\n"
    f"{CANONICAL_SCHEMA_DESC}\n\n"
    "For category normalization, prefer exact matches to known categories when obvious; set category_confidence accordingly.\n"
    "If ambiguous, return category with low confidence. Use empty tags for fields you cannot determine.\n"
    "When the user explicitly includes a numeric 'top N', include <n>. When they use plotting words ('show','plot','chart'), set <plot>true</plot>.\n"
    "Do not include additional commentary — output only the single-line XML block."
)

def build_parser_prompt(user_query: str, known_categories: list = None) -> str:
    known_text = ", ".join(known_categories[:60]) if known_categories else "none"
    ex_lines = []
    for ex in EXAMPLES:
        ex_lines.append(f"Q: {ex['query']}")
        ex_lines.append(f"A: {ex['xml']}")
    ex_block = "\n".join(ex_lines)
    prompt = (
        SYSTEM_INSTRUCTION
        + "\n\n"
        + "Known categories (helpful hint): " + known_text + "\n\n"
        + "Many-shot examples (do not mimic wording exactly, follow schema):\n"
        + ex_block + "\n\n"
        + "Now parse this user query and output ONLY the XML block on a single line:\n"
        + f"Q: {user_query}\n"
        + "A:"
    )
    return prompt
