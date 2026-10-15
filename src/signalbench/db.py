"""DuckDB storage. All public timestamps are timezone aware."""

from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS labeled_text (
 source VARCHAR, id VARCHAR PRIMARY KEY, text VARCHAR, label VARCHAR, split VARCHAR);
CREATE TABLE IF NOT EXISTS filings (
 cik VARCHAR, ticker VARCHAR, accession VARCHAR PRIMARY KEY,
 accepted_at TIMESTAMPTZ, items VARCHAR, url VARCHAR);
CREATE TABLE IF NOT EXISTS filing_text (accession VARCHAR PRIMARY KEY, text VARCHAR);
CREATE TABLE IF NOT EXISTS filing_scores (
 accession VARCHAR, model VARCHAR, score DOUBLE, fraction_positive DOUBLE,
 fraction_negative DOUBLE, sentences INTEGER, PRIMARY KEY(accession, model));
CREATE TABLE IF NOT EXISTS prices (
 ticker VARCHAR, date DATE, open DOUBLE, close DOUBLE, adj_close DOUBLE,
 volume DOUBLE, PRIMARY KEY(ticker, date));
CREATE TABLE IF NOT EXISTS failures (
 accession VARCHAR, stage VARCHAR, reason VARCHAR, PRIMARY KEY(accession, stage));
"""


def connect(path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open a database and create the idempotent schema."""
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(path)
    con.execute(SCHEMA)
    return con


def upsert(con: duckdb.DuckDBPyConnection, table: str, frame) -> None:
    """Insert named columns using primary-key replacement."""
    if table not in {
        "labeled_text",
        "filings",
        "filing_text",
        "filing_scores",
        "prices",
        "failures",
    }:
        raise ValueError("Unknown table")
    con.register("incoming", frame)
    try:
        con.execute(f"INSERT OR REPLACE INTO {table} BY NAME SELECT * FROM incoming")
    finally:
        con.unregister("incoming")
