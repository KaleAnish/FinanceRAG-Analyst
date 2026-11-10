from datetime import date

from finance_rag.query_understanding import detect_companies, detect_time, plan_query

TODAY = date(2026, 11, 10)
CORPUS = ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "T", "BA", "LMT",
          "JPM", "BAC", "GS", "MS", "BLK", "XOM", "DIS", "HD", "MCD", "PEP", "COST"]


def test_company_names_map_to_tickers_in_order():
    tickers, _ = detect_companies("Compare JPMorgan and Bank of America's net interest income", CORPUS)
    assert tickers == ["JPM", "BAC"]


def test_possessive_and_curly_apostrophe():
    tickers, _ = detect_companies("what does Boeing’s 10-K say about its 737s?", CORPUS)
    assert tickers == ["BA"]


def test_explicit_tickers_and_cashtags():
    tickers, _ = detect_companies("NVDA vs $AMD data center growth", CORPUS)
    assert tickers == ["NVDA", "AMD"]


def test_acronyms_are_not_tickers():
    tickers, _ = detect_companies("How is AI changing CEO pay and EPS at US banks?", CORPUS)
    assert tickers == []


def test_word_boundaries():
    tickers, _ = detect_companies("metadata and intelligence in applesauce", CORPUS)
    assert tickers == []


def test_longest_alias_wins():
    tickers, _ = detect_companies("JPMorgan Chase and Goldman Sachs", CORPUS)
    assert tickers == ["JPM", "GS"]


def test_att_single_letter_ticker_by_name_only():
    tickers, _ = detect_companies("What did AT&T say about fiber?", CORPUS)
    assert tickers == ["T"]


def test_uncovered_company_is_reported():
    tickers, uncovered = detect_companies("How is Delta compared with Boeing? and Tesla?", CORPUS)
    assert tickers == ["BA"]
    assert uncovered == ["tesla"]


def test_this_year():
    t = detect_time("what has Boeing been up to this year?", TODAY)
    assert (t["start"], t["end"], t["recency"]) == ("2026-01-01", "2026-11-10", True)


def test_last_quarter():
    t = detect_time("What was NVIDIA's data center revenue last quarter?", TODAY)
    assert t["start"] == "2026-06-13" and t["recency"] is True


def test_single_year_includes_annual_report_season():
    t = detect_time("Apple revenue in 2024", TODAY)
    assert (t["start"], t["end"]) == ("2024-01-01", "2025-03-31")


def test_year_range():
    t = detect_time("how did margins change between 2023 and 2025", TODAY)
    assert (t["start"], t["end"]) == ("2023-01-01", "2026-03-31")


def test_since_year():
    t = detect_time("buybacks since 2024", TODAY)
    assert (t["start"], t["end"]) == ("2024-01-01", "2026-11-10")


def test_quarter_and_year():
    t = detect_time("JPM results in Q3 2025", TODAY)
    assert t["start"] == "2025-07-01" and t["end"] == "2026-01-28"


def test_fiscal_year():
    t = detect_time("NVIDIA fiscal 2026 data center revenue", TODAY)
    assert (t["start"], t["end"]) == ("2025-01-01", "2026-11-10")


def test_rolling_window():
    t = detect_time("what has Meta said about capex in the past 18 months", TODAY)
    assert t["start"] == "2025-05-10"


def test_compared_to_last_year_is_not_a_window():
    t = detect_time("how did revenue change compared to last year?", TODAY)
    assert t["start"] is None and t["recency"] is True


def test_model_numbers_are_not_years():
    t = detect_time("what does boeing say about the 737-10 and 777X?", TODAY)
    assert t["start"] is None


def test_no_time_phrase_prefers_recent():
    t = detect_time("what supply chain risks does apple disclose?", TODAY)
    assert t == {"start": None, "end": None, "recency": True, "label": None}


def test_plan_comparison():
    p = plan_query("compare the drivers of data center revenue growth at nvidia and amd.", CORPUS, TODAY)
    assert p.tickers == ["NVDA", "AMD"] and p.intent == "comparison"
    assert p.preferred_forms == ["10-Q", "10-K"]


def test_plan_summary():
    p = plan_query("What has Boeing been up to this year?", CORPUS, TODAY)
    assert p.tickers == ["BA"] and p.intent == "summary" and p.time_label == "2026 so far"


def test_plan_numeric():
    p = plan_query("What was NVIDIA's data center revenue last quarter?", CORPUS, TODAY)
    assert p.intent == "numeric" and p.preferred_forms == ["10-Q", "10-K"]


def test_plan_event():
    p = plan_query("Did Goldman announce any layoffs?", CORPUS, TODAY)
    assert p.tickers == ["GS"] and p.intent == "event" and p.preferred_forms == ["8-K"]


def test_plan_lookup_without_company():
    p = plan_query("which companies mention tariffs as a risk?", CORPUS, TODAY)
    assert p.tickers == [] and p.intent == "lookup"


def test_lowercase_tickers_only_when_safe():
    tickers, _ = detect_companies("nvda vs amd, and what does it cost? ms word dis", CORPUS)
    assert tickers == ["NVDA", "AMD"]


def _evidence(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["chunk_id", "ticker", "filing_date", "rank"])


def test_recency_promotes_newer_filing():
    from finance_rag.query_understanding import rerank_by_recency
    ev = _evidence([("old", "BA", "2024-01-31", 1)] + [(f"x{i}", "BA", "2023-01-01", i + 2) for i in range(8)]
                   + [("new", "BA", "2026-07-29", 10)])
    out = rerank_by_recency(ev, top_k=5, today=TODAY)
    assert out["chunk_id"].iloc[0] == "new"
    assert list(out["rank"]) == [1, 2, 3, 4, 5]


def test_recency_keeps_strong_relevance_for_same_age():
    from finance_rag.query_understanding import rerank_by_recency
    ev = _evidence([("a", "BA", "2026-01-30", 1), ("b", "BA", "2026-01-30", 2)])
    out = rerank_by_recency(ev, top_k=2, today=TODAY)
    assert list(out["chunk_id"]) == ["a", "b"]


def test_recency_alternates_companies():
    from finance_rag.query_understanding import rerank_by_recency
    ev = _evidence([(f"n{i}", "NVDA", "2026-08-26", i + 1) for i in range(6)]
                   + [("a1", "AMD", "2025-08-06", 7), ("a2", "AMD", "2026-08-05", 8)])
    out = rerank_by_recency(ev, top_k=5, today=TODAY)
    assert list(out["ticker"]) == ["NVDA", "AMD", "NVDA", "AMD", "NVDA"]
    assert out["chunk_id"].iloc[1] == "a2"


def test_recency_handles_empty():
    from finance_rag.query_understanding import rerank_by_recency
    import pandas as pd
    assert rerank_by_recency(pd.DataFrame(), top_k=5) is not None


def test_interleave_pools_alternates_and_dedupes():
    from finance_rag.query_understanding import interleave_pools
    main = _evidence([("m1", "NVDA", "2024-05-29", 1), ("m2", "NVDA", "2024-05-29", 2), ("dup", "NVDA", "2026-08-26", 3)])
    fresh = _evidence([("dup", "NVDA", "2026-08-26", 1), ("f2", "NVDA", "2026-05-20", 2)])
    out = interleave_pools(main, fresh)
    assert list(out["chunk_id"]) == ["m1", "dup", "m2", "f2"]


def test_interleave_then_rerank_puts_recent_first():
    from finance_rag.query_understanding import interleave_pools, rerank_by_recency
    main = _evidence([(f"old{i}", "NVDA", "2024-05-29", i + 1) for i in range(5)])
    fresh = _evidence([("new1", "NVDA", "2026-08-26", 1)])
    out = rerank_by_recency(interleave_pools(main, fresh), top_k=3, today=TODAY)
    assert out["chunk_id"].iloc[0] == "new1"


def test_interleave_handles_empty():
    from finance_rag.query_understanding import interleave_pools
    main = _evidence([("m1", "BA", "2026-01-30", 1)])
    assert list(interleave_pools(main, None)["chunk_id"]) == ["m1"]
    assert list(interleave_pools(None, main)["chunk_id"]) == ["m1"]
