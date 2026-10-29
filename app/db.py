"""SQLite access for ResolveBot.

Every query is parameterised. Every order query is scoped to a customer_id that
comes from the authenticated ``X-Customer-Id`` header, never from the LLM.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.config import get_settings


@contextmanager
def connection(write: bool = False) -> Iterator[sqlite3.Connection]:
    """Open a connection. With ``write=True`` the body runs in one
    ``BEGIN IMMEDIATE`` transaction that commits on success and rolls back on
    any exception."""
    conn = sqlite3.connect(get_settings().db_path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            if conn.in_transaction:
                conn.execute("COMMIT")
    finally:
        conn.close()


def customer_exists(customer_id: str) -> bool:
    with connection() as conn:
        return conn.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone() is not None


def fetch_policy_parameters(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute(
        "SELECT key, value, unit, source_doc_id, source_section FROM policy_parameters"
    ).fetchall()
    return {row["key"]: dict(row) for row in rows}


def fetch_customer_order(conn: sqlite3.Connection, customer_id: str, order_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
        (order_id, customer_id),
    ).fetchone()
    return dict(row) if row else None


def fetch_customer_orders(
    conn: sqlite3.Connection, customer_id: str, status: str | None = None, limit: int = 25
) -> list[dict]:
    sql = "SELECT * FROM orders WHERE customer_id = ?"
    params: list = [customer_id]
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY order_date DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def fetch_order_items(conn: sqlite3.Connection, order_id: str) -> list[dict]:
    """Items of an order. Callers must have verified ownership of ``order_id``."""
    rows = conn.execute(
        """
        SELECT oi.item_id, oi.order_id, oi.product_id, oi.qty, oi.unit_price,
               oi.opened, p.name, p.category, p.returnable, p.warranty_class
        FROM order_items AS oi
        JOIN products AS p ON p.product_id = oi.product_id
        WHERE oi.order_id = ?
        ORDER BY oi.item_id
        """,
        (order_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def fetch_shipment(conn: sqlite3.Connection, order_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM shipments WHERE order_id = ?", (order_id,)
    ).fetchone()
    return dict(row) if row else None
