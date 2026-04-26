from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_symbol(value: Any) -> str:
    symbol = clean_text(value).upper()
    return re.sub(r"[^A-Z0-9.\-=]", "", symbol)


def normalise_quote(raw: dict[str, Any]) -> dict[str, str]:
    symbol = clean_symbol(raw.get("symbol"))
    name = (
        clean_text(raw.get("shortname"))
        or clean_text(raw.get("longname"))
        or clean_text(raw.get("name"))
        or symbol
    )

    return {
        "symbol": symbol,
        "name": name,
        "exchange": clean_text(raw.get("exchange")),
        "type": clean_text(raw.get("quoteType") or raw.get("typeDisp")),
    }


def run_search(query: str, limit: int) -> list[dict[str, str]]:
    import yfinance as yf
    import yfinance.cache as yf_cache

    cache_dir = Path(__file__).resolve().parents[1] / "storage" / "yfinance-cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_dir = Path(tempfile.gettempdir()) / "cs-project-yfinance-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

    yf_cache.set_cache_location(str(cache_dir))
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))

    quotes: list[dict[str, Any]] = []

    if hasattr(yf, "Search"):
        search = yf.Search(query, max_results=limit)
        quotes = list(getattr(search, "quotes", []) or [])
    elif hasattr(yf, "lookup"):
        result = yf.lookup(query)
        if hasattr(result, "to_dict"):
            quotes = list(result.to_dict(orient="records"))

    results: list[dict[str, str]] = []
    seen: set[str] = set()

    for quote in quotes:
        if not isinstance(quote, dict):
            continue

        item = normalise_quote(quote)
        symbol = item["symbol"]
        if symbol == "" or symbol in seen:
            continue

        seen.add(symbol)
        results.append(item)

        if len(results) >= limit:
            break

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Search Yahoo Finance tickers through yfinance.")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    query = args.query.strip()
    if query == "":
        print(json.dumps({"ok": True, "results": []}, separators=(",", ":")))
        return 0

    try:
        results = run_search(query, max(1, min(args.limit, 25)))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "results": []}, separators=(",", ":")))
        return 0

    print(json.dumps({"ok": True, "results": results}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
