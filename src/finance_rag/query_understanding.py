"""Plain-English query understanding (Layer B1, part 1).

Turns a question into a QueryPlan: which companies it names, what time
window it implies, and what kind of question it is. Everything here is
plain Python (no model calls), so it is fast, deterministic and testable.
The service layer decides how to apply the plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Optional

# Names and nicknames people actually type, mapped to tickers. Only tickers
# that exist in the loaded corpus are used; the rest are reported as
# "uncovered" so the app can say so instead of silently searching everything.
COMPANY_ALIASES: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "broadcom": "AVGO",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "advanced micro devices": "AMD",
    "adobe": "ADBE",
    "intel": "INTC",
    "cisco": "CSCO",
    "alphabet": "GOOGL", "google": "GOOGL", "youtube": "GOOGL",
    "meta platforms": "META", "meta": "META", "facebook": "META",
    "instagram": "META", "whatsapp": "META",
    "netflix": "NFLX",
    "walt disney": "DIS", "disney": "DIS",
    "at&t": "T",
    "verizon": "VZ",
    "amazon": "AMZN", "aws": "AMZN",
    "tesla": "TSLA",
    "home depot": "HD",
    "mcdonald's": "MCD", "mcdonalds": "MCD",
    "nike": "NKE",
    "starbucks": "SBUX",
    "costco": "COST",
    "walmart": "WMT",
    "procter & gamble": "PG", "procter and gamble": "PG", "p&g": "PG",
    "coca-cola": "KO", "coca cola": "KO",
    "pepsico": "PEP", "pepsi": "PEP",
    "jpmorgan chase": "JPM", "jpmorgan": "JPM", "jp morgan": "JPM", "jpmc": "JPM",
    "bank of america": "BAC", "bofa": "BAC",
    "goldman sachs": "GS", "goldman": "GS",
    "morgan stanley": "MS",
    "blackrock": "BLK",
    "wells fargo": "WFC",
    "citigroup": "C", "citibank": "C",
    "mastercard": "MA",
    "berkshire hathaway": "BRK.B", "berkshire": "BRK.B",
    "unitedhealth": "UNH",
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ",
    "eli lilly": "LLY",
    "pfizer": "PFE",
    "merck": "MRK",
    "abbvie": "ABBV",
    "boeing": "BA",
    "lockheed martin": "LMT", "lockheed": "LMT",
    "caterpillar": "CAT",
    "general electric": "GE", "ge aerospace": "GE",
    "honeywell": "HON",
    "raytheon": "RTX",
    "exxonmobil": "XOM", "exxon mobil": "XOM", "exxon": "XOM",
    "chevron": "CVX",
}

# Tickers that are safe to recognise even when typed in lower case ("nvda",
# "amd"). Tickers that are also ordinary words (cost, dis, ms, ba, hd, t, v,
# c, ma, ko, pg, ge, cat, hon) are excluded on purpose.
_LOWERCASE_SAFE_TICKERS = {
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "INTC", "CSCO",
    "GOOGL", "GOOG", "META", "NFLX", "AMZN", "TSLA", "MCD", "NKE", "SBUX", "WMT",
    "PEP", "JPM", "BAC", "BLK", "WFC", "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
    "LMT", "RTX", "XOM", "CVX", "VZ",
}
for _t in _LOWERCASE_SAFE_TICKERS:
    COMPANY_ALIASES.setdefault(_t.lower(), _t)

# Upper-case words that look like tickers but usually aren't meant as one.
_NOT_TICKERS = {
    "AI", "CEO", "CFO", "COO", "CTO", "US", "USA", "UK", "EU", "EPS", "GDP",
    "FY", "SEC", "IPO", "ETF", "ESG", "API", "LLM", "GPU", "CPU", "YOY",
    "QOQ", "YTD", "TTM", "M&A", "R&D", "NII", "EBIT", "EBITDA", "FCF",
}

_COMPARE = re.compile(
    r"\b(compare|compared|comparison|versus|vs\.?|relative to|differ|difference|"
    r"against|better than|worse than|outperform|which (?:company|bank|one|of))\b"
)
_SUMMARY = re.compile(
    r"(\bup to\b|\boverview\b|\bsummar|\bhighlights?\b|what'?s new|what is new|"
    r"what happened|recent developments|key (?:events|developments|updates)|"
    r"tell me about|update on|\bnews\b|been doing)"
)
_EVENT = re.compile(
    r"\b(announce\w*|acquisitions?|acquire[sd]?|mergers?|deals?|ceo|cfo|resign\w*|"
    r"appoint\w*|hire[sd]?|layoffs?|lawsuits?|settle\w*|orders?|contracts?|"
    r"dividends?|buybacks?|repurchases?|restructuring)\b"
)
_METRIC = re.compile(
    r"\b(revenue|revenues|sales|income|profit|profits|loss|losses|eps|earnings|"
    r"margin|margins|cash flow|cash|debt|backlog|guidance|capex|capital expenditures?|"
    r"spending|expenses|net interest income|deposits|assets|valuation|market cap\w*)\b"
)
_NUMERIC_ASK = re.compile(r"\b(how much|how many|what (?:was|is|were|are)|growth|grew|increase|decrease|change)\b")

_MONTHS_FOR_ANNUAL_REPORT = 3  # 10-Ks for a year are usually filed by the end of March


@dataclass
class QueryPlan:
    tickers: list[str] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)  # named companies not in the corpus
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    recency: bool = True  # prefer newer filings when ranking
    time_label: Optional[str] = None
    intent: str = "lookup"  # lookup | comparison | summary | numeric | event
    preferred_forms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- companies

def _normalize(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def detect_companies(question: str, available: Optional[Iterable[str]] = None) -> tuple[list[str], list[str]]:
    """Return (tickers in corpus, named companies/tickers not in corpus), in order of appearance."""
    available_set = {t.upper() for t in available} if available is not None else None
    text = _normalize(question)
    lowered = text.lower()
    hits: list[tuple[int, str, str]] = []  # (position, ticker, surface text)

    # Company names, longest first, masking each match so "jpmorgan chase" isn't counted twice.
    masked = lowered
    for alias in sorted(COMPANY_ALIASES, key=len, reverse=True):
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])")
        for m in pattern.finditer(masked):
            hits.append((m.start(), COMPANY_ALIASES[alias], alias))
            masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end():]

    # Explicit tickers: upper-case tokens (2+ letters) or cashtags like $NVDA.
    for m in re.finditer(r"(?<![A-Za-z0-9])\$?([A-Z]{2,5}(?:\.[A-Z])?)(?![A-Za-z0-9&])", text):
        token = m.group(1)
        if token in _NOT_TICKERS:
            continue
        if available_set is not None and token not in available_set:
            continue
        if available_set is None and token not in set(COMPANY_ALIASES.values()):
            continue
        hits.append((m.start(1), token, token))

    tickers: list[str] = []
    uncovered: list[str] = []
    for _, ticker, surface in sorted(hits):
        if available_set is not None and ticker not in available_set:
            if surface not in uncovered:
                uncovered.append(surface)
            continue
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers, uncovered


# ---------------------------------------------------------------- time

def _end_of_annual_report_season(year: int) -> date:
    return date(year + 1, _MONTHS_FOR_ANNUAL_REPORT, 31)


def _cap(d: date, today: date) -> date:
    return min(d, today)


def _months_ago(today: date, months: int) -> date:
    y, m = today.year, today.month - months
    while m <= 0:
        m += 12
        y -= 1
    day = min(today.day, 28)
    return date(y, m, day)


def detect_time(question: str, today: date) -> dict:
    """Return start/end (ISO strings or None), recency flag and a short label."""
    q = _normalize(question).lower()

    def window(start: date, end: date, label: str, recency: bool = False) -> dict:
        return {"start": start.isoformat(), "end": _cap(end, today).isoformat(),
                "recency": recency, "label": label}

    # Year ranges: "between 2023 and 2025", "from 2023 to 2025", "2023-2025"
    m = re.search(r"\b(?:between|from)?\s*(20\d{2})\s*(?:and|to|through|-|–)\s*(20\d{2})\b", q)
    if m and int(m.group(1)) <= int(m.group(2)):
        a, b = int(m.group(1)), int(m.group(2))
        return window(date(a, 1, 1), _end_of_annual_report_season(b), f"{a}–{b}")

    m = re.search(r"\bsince\s+(20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        return window(date(y, 1, 1), today, f"since {y}", recency=True)

    # Quarters: "Q3 2025", "q3 of 2025"
    m = re.search(r"\bq([1-4])\s*(?:of\s+)?(?:fy\s*)?(20\d{2})\b", q)
    if m:
        qn, y = int(m.group(1)), int(m.group(2))
        q_start = date(y, 3 * (qn - 1) + 1, 1)
        q_end = date(y + (1 if qn == 4 else 0), 1 if qn == 4 else 3 * qn + 1, 1) - timedelta(days=1)
        return window(q_start, q_end + timedelta(days=120), f"Q{qn} {y}")

    # Fiscal years: "fiscal 2025", "FY2025", "FY25"
    m = re.search(r"\b(?:fiscal(?: year)?\s*|fy\s*)(20\d{2}|\d{2})\b", q)
    if m:
        y = int(m.group(1))
        y = y + 2000 if y < 100 else y
        return window(date(y - 1, 1, 1), _end_of_annual_report_season(y), f"fiscal {y}")

    # Rolling windows: "past 18 months", "last 2 years", "last 3 quarters"
    m = re.search(r"\b(?:past|last|previous)\s+(\d{1,2})\s+(month|year|quarter)s?\b", q)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        months = n if unit == "month" else n * 12 if unit == "year" else n * 3
        return window(_months_ago(today, months), today, f"past {n} {unit}s", recency=True)

    if re.search(r"\b(this year|so far this year|year to date|ytd)\b", q):
        return window(date(today.year, 1, 1), today, f"{today.year} so far", recency=True)

    m = re.search(r"\blast year\b", q)
    if m:
        before = q[: m.start()].rstrip()
        if re.search(r"\b(from|than|vs\.?|versus|compared to|compared with|over|since)$", before):
            return {"start": None, "end": None, "recency": True, "label": "latest vs last year"}
        y = today.year - 1
        return window(date(y, 1, 1), _end_of_annual_report_season(y), str(y))

    if re.search(r"\b(last|latest|most recent|recent|this|previous|past) (quarter|10-?q)\b", q) or "last quarter" in q:
        return window(today - timedelta(days=150), today, "latest quarter", recency=True)

    if re.search(r"\b(this|last|past) month\b", q):
        return window(today - timedelta(days=60), today, "past 2 months", recency=True)

    # Single years: "in 2025", "2024 revenue"
    m = re.search(r"(?<![\d-])(20\d{2})(?![\d-])", q)
    if m:
        y = int(m.group(1))
        if 2000 <= y <= today.year:
            return window(date(y, 1, 1), _end_of_annual_report_season(y), str(y))

    if re.search(r"\b(latest|most recent|recent|recently|lately|newest)\b", q):
        return window(today - timedelta(days=365), today, "past 12 months", recency=True)

    # Nothing explicit: no hard window, but prefer newer filings.
    return {"start": None, "end": None, "recency": True, "label": None}


# ---------------------------------------------------------------- intent

def detect_intent(question: str, n_tickers: int) -> tuple[str, list[str]]:
    q = _normalize(question).lower()
    if n_tickers >= 2 or _COMPARE.search(q):
        intent = "comparison"
    elif _SUMMARY.search(q):
        intent = "summary"
    elif _METRIC.search(q) and _NUMERIC_ASK.search(q):
        intent = "numeric"
    elif _EVENT.search(q):
        intent = "event"
    else:
        intent = "lookup"

    if intent == "event":
        forms = ["8-K"]
    elif intent == "numeric" or (intent == "comparison" and _METRIC.search(q)):
        forms = ["10-Q", "10-K"]
    else:
        forms = []
    return intent, forms


# ---------------------------------------------------------------- plan

def plan_query(question: str, available_tickers: Optional[Iterable[str]] = None,
               today: Optional[date] = None) -> QueryPlan:
    today = today or date.today()
    tickers, uncovered = detect_companies(question, available_tickers)
    t = detect_time(question, today)
    intent, forms = detect_intent(question, len(tickers))

    plan = QueryPlan(
        tickers=tickers,
        uncovered=uncovered,
        start_date=t["start"],
        end_date=t["end"],
        recency=t["recency"],
        time_label=t["label"],
        intent=intent,
        preferred_forms=forms,
    )
    if tickers:
        plan.notes.append("companies: " + ", ".join(tickers))
    if uncovered:
        plan.notes.append("not covered: " + ", ".join(uncovered))
    if t["label"]:
        plan.notes.append("time: " + t["label"])
    plan.notes.append("intent: " + intent)
    return plan


# ---------------------------------------------------------------- recency

def rerank_by_recency(evidence, top_k: int, today: Optional[date] = None,
                      half_life_days: int = 1095, rrf_k: int = 30):
    """Re-rank retrieved passages so newer filings win unless an older one is
    clearly more relevant.

    Relevance is taken from the incoming order (1 / (rrf_k + position)), then
    multiplied by 0.5 ** (age / half_life). With a 3-year half-life, a 2026
    filing at position 10 beats a 2024 filing at position 1. With several
    companies, each company is ranked separately and the results alternate,
    so a comparison never loses one side.
    """
    import pandas as pd

    if evidence is None or len(evidence) == 0:
        return evidence
    today = today or date.today()
    df = evidence.reset_index(drop=True).copy()
    ages = (pd.Timestamp(today) - pd.to_datetime(df["filing_date"])).dt.days.clip(lower=0).to_numpy()
    relevance = 1.0 / (rrf_k + df.index.to_numpy() + 1)
    df["_score"] = relevance * (0.5 ** (ages / half_life_days))

    if "ticker" in df.columns and df["ticker"].nunique() > 1:
        groups = [g.sort_values("_score", ascending=False, kind="stable")
                  for _, g in df.groupby("ticker", sort=False)]
        order, i = [], 0
        while len(order) < len(df):
            for g in groups:
                if i < len(g):
                    order.append(g.index[i])
            i += 1
        df = df.loc[order]
    else:
        df = df.sort_values("_score", ascending=False, kind="stable")

    df = df.head(top_k).drop(columns="_score").reset_index(drop=True)
    if "rank" in df.columns:
        df["rank"] = range(1, len(df) + 1)
    return df


def fresh_window_start(today: Optional[date] = None, days: int = 365) -> str:
    """Start date (ISO) of the 'recent filings' pool."""
    return ((today or date.today()) - timedelta(days=days)).isoformat()


def interleave_pools(main, fresh):
    """Merge the main results with a recent-filings pool, alternating rows so
    recent passages get a fair position, and dropping duplicate passages."""
    import pandas as pd

    if fresh is None or len(fresh) == 0:
        return main
    if main is None or len(main) == 0:
        return fresh
    a, b = main.reset_index(drop=True), fresh.reset_index(drop=True)
    rows = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            rows.append(a.iloc[i])
        if i < len(b):
            rows.append(b.iloc[i])
    merged = pd.DataFrame(rows).reset_index(drop=True)
    key = "chunk_id" if "chunk_id" in merged.columns else None
    if key:
        merged = merged.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------- boilerplate

_BOILERPLATE_PATTERNS = [
    re.compile(r"(?<![\d.])(?:10|31|32|101|104)\.\d{1,3}(?:\.\d+)?(?![\d.])"),  # exhibit numbers
    re.compile(r"sarbanes-oxley", re.I),
    re.compile(r"inline xbrl", re.I),
    re.compile(r"\bcertification of (?:chief|principal)", re.I),
    re.compile(r"\bexhibit \d", re.I),
    re.compile(r"pursuant to the requirements of the securities exchange act", re.I),
    re.compile(r"indicate by check mark", re.I),
    re.compile(r"commission file number", re.I),
    re.compile(r"\b(?:incorporated (?:herein )?by reference)\b", re.I),
    re.compile(r"\binteractive data file\b", re.I),
]


def boilerplate_score(text: str) -> int:
    """How strongly a passage looks like an exhibit index, signature page,
    certification list or cover page (pattern hits, counted)."""
    if not isinstance(text, str):
        return 0
    return sum(len(p.findall(text)) for p in _BOILERPLATE_PATTERNS)


def drop_boilerplate(evidence, threshold: int = 6):
    """Remove boilerplate passages; keep everything if that would empty the list."""
    if evidence is None or len(evidence) == 0 or "text" not in evidence.columns:
        return evidence
    keep = evidence["text"].map(boilerplate_score) < threshold
    if not keep.any():
        return evidence
    return evidence[keep].reset_index(drop=True)
