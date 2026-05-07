"""
SQL Integration Module
=======================
Stores cleaned IPL Auction dataset in SQLite and provides
analytical query functions for the Streamlit dashboard.
"""

import sqlite3
import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = os.environ.get("DB_PATH", "data/ipl_auction.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_to_db(df: pd.DataFrame, table: str = "ipl_auction") -> None:
    """Write cleaned DataFrame to SQLite, replacing existing data."""
    conn = get_connection()
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    logger.info(f"Loaded {len(df)} rows into SQLite table '{table}' at {DB_PATH}")


def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    try:
        result = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built SQL Reports
# ─────────────────────────────────────────────────────────────────────────────

SQL_REPORTS = {
    "team_total_spend": """
        SELECT
            current_team AS team,
            ROUND(SUM(cost_cr), 2) AS total_spend_cr,
            COUNT(*) AS players_bought,
            ROUND(AVG(cost_cr), 2) AS avg_cost_cr,
            ROUND(MAX(cost_cr), 2) AS max_cost_cr
        FROM ipl_auction
        WHERE is_sold = 1 AND is_retained = 0 AND cost_cr > 0
        GROUP BY current_team
        ORDER BY total_spend_cr DESC
    """,

    "top_10_expensive": """
        SELECT
            player_name, current_team, player_role,
            player_origin, cost_cr, base_price_cr,
            ROUND(cost_cr / NULLIF(base_price_cr, 0), 1) AS multiplier
        FROM ipl_auction
        WHERE is_sold = 1 AND is_retained = 0 AND cost_cr > 0
        ORDER BY cost_cr DESC
        LIMIT 10
    """,

    "role_stats": """
        SELECT
            player_role,
            COUNT(*) AS total_players,
            SUM(CASE WHEN is_sold = 1 THEN 1 ELSE 0 END) AS sold,
            SUM(CASE WHEN is_sold = 0 THEN 1 ELSE 0 END) AS unsold,
            ROUND(SUM(cost_cr), 2) AS total_spend_cr,
            ROUND(AVG(CASE WHEN is_sold=1 AND cost_cr>0 THEN cost_cr END), 2) AS avg_price_cr
        FROM ipl_auction
        GROUP BY player_role
        ORDER BY total_spend_cr DESC
    """,

    "origin_summary": """
        SELECT
            player_origin,
            COUNT(*) AS total_in_pool,
            SUM(CASE WHEN is_sold = 1 THEN 1 ELSE 0 END) AS sold,
            ROUND(SUM(cost_cr), 2) AS total_spend_cr,
            ROUND(AVG(CASE WHEN is_sold=1 AND cost_cr>0 THEN cost_cr END), 2) AS avg_auction_price_cr
        FROM ipl_auction
        WHERE is_retained = 0
        GROUP BY player_origin
        ORDER BY total_spend_cr DESC
    """,

    "bargain_buys": """
        SELECT
            player_name, current_team, player_role,
            cost_cr, base_price_cr,
            ROUND(cost_cr / NULLIF(base_price_cr, 0), 1) AS multiplier
        FROM ipl_auction
        WHERE is_sold = 1 AND is_retained = 0
          AND cost_cr > 0 AND cost_cr <= 3
          AND base_price_cr > 0
        ORDER BY multiplier DESC
        LIMIT 15
    """,

    "team_foreign_count": """
        SELECT
            current_team AS team,
            SUM(CASE WHEN player_origin = 'Overseas' THEN 1 ELSE 0 END) AS overseas_players,
            SUM(CASE WHEN player_origin = 'Indian' THEN 1 ELSE 0 END) AS indian_players,
            COUNT(*) AS total_squad
        FROM ipl_auction
        WHERE is_sold = 1
        GROUP BY current_team
        ORDER BY overseas_players DESC
    """,

    "price_bucket_dist": """
        SELECT
            price_bucket,
            COUNT(*) AS player_count,
            ROUND(SUM(cost_cr), 2) AS total_spend_cr
        FROM ipl_auction
        WHERE is_sold = 1 AND is_retained = 0 AND cost_cr > 0
        GROUP BY price_bucket
        ORDER BY total_spend_cr DESC
    """,

    "team_switched_players": """
        SELECT
            player_name,
            prev_team_2022 AS old_team,
            current_team AS new_team,
            player_role,
            cost_cr
        FROM ipl_auction
        WHERE team_changed = 1 AND is_sold = 1
        ORDER BY cost_cr DESC
        LIMIT 20
    """,
}


def get_report(report_name: str) -> pd.DataFrame:
    """Run a named pre-built SQL report."""
    if report_name not in SQL_REPORTS:
        raise ValueError(f"Unknown report: {report_name}. Available: {list(SQL_REPORTS.keys())}")
    sql = SQL_REPORTS[report_name]
    result = run_query(sql)
    logger.info(f"Report '{report_name}': {len(result)} rows returned")
    return result


def search_player(name: str) -> pd.DataFrame:
    """Full-text search for a player by name (case-insensitive partial match)."""
    sql = """
        SELECT player_name, current_team, player_role, player_origin,
               cost_cr, base_price_cr, price_bucket, price_multiplier,
               prev_team_2022, is_sold, is_retained, team_changed
        FROM ipl_auction
        WHERE LOWER(player_name) LIKE LOWER(?)
        ORDER BY cost_cr DESC
    """
    return run_query(sql, (f"%{name}%",))


def get_team_players(team: str) -> pd.DataFrame:
    """Return all players for a given team."""
    sql = """
        SELECT player_name, player_role, player_origin,
               cost_cr, base_price_cr, price_bucket, prev_team_2022, is_retained
        FROM ipl_auction
        WHERE current_team = ?
        ORDER BY cost_cr DESC
    """
    return run_query(sql, (team,))


def setup_database(df: pd.DataFrame) -> None:
    """Full DB setup: load data and create views."""
    load_to_db(df)
    conn = get_connection()
    # Create a summary view
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_sold_summary AS
        SELECT * FROM ipl_auction WHERE is_sold = 1
    """)
    conn.commit()
    conn.close()
    logger.info("Database setup complete")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.cleaning.pipeline import run_pipeline
    df = run_pipeline(save=False)
    setup_database(df)

    print("\n=== TEAM TOTAL SPENDING ===")
    print(get_report("team_total_spend").to_string(index=False))

    print("\n=== TOP 10 MOST EXPENSIVE ===")
    print(get_report("top_10_expensive").to_string(index=False))

    print("\n=== ROLE STATS ===")
    print(get_report("role_stats").to_string(index=False))

    print("\n=== PLAYER SEARCH: 'Curran' ===")
    print(search_player("Curran").to_string(index=False))
