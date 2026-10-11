"""Optional customer memory (SQLite). Owner: P4.

Stores which insights were shown and how the customer responded. It may only RE-RANK, DEMOTE or
COOL OFF rows that P5's engine already marked eligible_to_show; it never changes propensity,
confidence, eligibility, impact, reason codes, evidence or P5's `rank` (brief, slide 15).
Every row gets row["memory"] = {"action", "reason", "display_rank", ...} so the UI can say why.

    action        SHOW | BOOST | DEMOTE | COOL_OFF | HIDE
    reason        NO_MEMORY_EFFECT | RECENT_INTEREST | SHOWN_WITHOUT_ENGAGEMENT | RETURNING_AFTER_COOL_OFF
                  | DISMISSED_COOL_OFF | CUSTOMER_SAID_NOT_RELEVANT | ALREADY_TOOK_UP | NOT_ELIGIBLE
    display_rank  order to display after memory (None if hidden); P5's `rank` is left untouched
"""
from __future__ import annotations

import copy
import sqlite3
from threading import RLock
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Literal

from src.common.config import DATA_DIR

Event = Literal["shown", "clicked", "dismissed", "not_relevant", "took_up"]
EVENTS = {"shown", "clicked", "dismissed", "not_relevant", "took_up"}
DB_PATH = DATA_DIR / "memory" / "customer_memory.db"  # git-ignored

COOL_OFF_DAYS = {"dismissed": 30, "dismissed_twice": 90, "not_relevant": 180}
FATIGUE_SHOWS = 3           # shown this many times since the last click -> demote
FATIGUE_MULTIPLIER = 0.5
INTEREST_MULTIPLIER = 1.25  # clicked recently -> keep near the top for follow-up
INTEREST_WINDOW_DAYS = 30
PROTECTED_FIELDS = ("propensity", "confidence", "eligible_to_show", "impact", "reason_codes",
                    "evidence", "rank", "ranking_score", "suppression_reason")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    product_code TEXT NOT NULL,
    event TEXT NOT NULL,
    on_date TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_customer ON events(customer_id, on_date);
"""


class MemoryStore:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def record(self, customer_id: str, product_code: str, event: Event, on: date) -> None:
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}; expected one of {sorted(EVENTS)}")
        with self.lock, self.conn:
            self.conn.execute("INSERT INTO events (customer_id, product_code, event, on_date) VALUES (?,?,?,?)",
                              (customer_id, product_code, event, on.isoformat()))

    def history(self, customer_id: str) -> list[dict]:
        with self.lock:
            cur = self.conn.execute("SELECT product_code, event, on_date FROM events WHERE customer_id=? ORDER BY on_date, id",
                                    (customer_id,))
            return [dict(r) for r in cur.fetchall()]

    def stamp(self) -> tuple[int, int]:
        """(latest event id, event count): changes whenever any process records or forgets
        feedback, so cached portfolio aggregates know when to recompute."""
        with self.lock:
            row = self.conn.execute("SELECT COALESCE(MAX(id), 0), COUNT(*) FROM events").fetchone()
            return int(row[0]), int(row[1])

    def event_counts(self) -> dict[str, dict[str, int]]:
        """{product_code: {event: distinct customers}} across all stored feedback."""
        with self.lock:
            rows = self.conn.execute("SELECT product_code, event, COUNT(DISTINCT customer_id) FROM events "
                                     "GROUP BY product_code, event").fetchall()
        out: dict[str, dict[str, int]] = {}
        for code, event, n in rows:
            out.setdefault(code, {})[event] = int(n)
        return out

    def forget(self, customer_id: str) -> int:
        """Right to erasure: delete everything stored for a customer."""
        with self.lock, self.conn:
            return self.conn.execute("DELETE FROM events WHERE customer_id=?", (customer_id,)).rowcount


_default_store: MemoryStore | None = None


def default_store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store


def _state(history: Iterable[dict], product_code: str) -> dict:
    events = [e for e in history if e["product_code"] == product_code]
    last = lambda kind: max((e["on_date"] for e in events if e["event"] == kind), default=None)  # noqa: E731
    last_click = last("clicked")
    return {
        "times_shown": sum(e["event"] == "shown" for e in events),
        "shows_since_click": sum(e["event"] == "shown" and (last_click is None or e["on_date"] > last_click)
                                 for e in events),
        "dismissals": sum(e["event"] == "dismissed" for e in events),
        "last_dismissed": last("dismissed"),
        "last_not_relevant": last("not_relevant"),
        "last_clicked": last_click,
        "took_up": any(e["event"] == "took_up" for e in events),
    }


def _days(since: str | None, today: date) -> float:
    return float("inf") if since is None else (today - date.fromisoformat(since)).days


def decide(row: dict, state: dict, today: date) -> tuple[str, str, float]:
    """Return (action, reason, multiplier)."""
    if not row.get("eligible_to_show"):
        return "HIDE", "NOT_ELIGIBLE", 0.0  # memory never overrides P5's eligibility
    if state["took_up"]:
        return "HIDE", "ALREADY_TOOK_UP", 0.0
    if _days(state["last_not_relevant"], today) < COOL_OFF_DAYS["not_relevant"]:
        return "COOL_OFF", "CUSTOMER_SAID_NOT_RELEVANT", 0.0
    window = COOL_OFF_DAYS["dismissed_twice"] if state["dismissals"] >= 2 else COOL_OFF_DAYS["dismissed"]
    if _days(state["last_dismissed"], today) < window:
        return "COOL_OFF", "DISMISSED_COOL_OFF", 0.0
    if _days(state["last_clicked"], today) < INTEREST_WINDOW_DAYS:
        return "BOOST", "RECENT_INTEREST", INTEREST_MULTIPLIER
    if state["shows_since_click"] >= FATIGUE_SHOWS:
        return "DEMOTE", "SHOWN_WITHOUT_ENGAGEMENT", FATIGUE_MULTIPLIER
    if state["dismissals"]:
        return "SHOW", "RETURNING_AFTER_COOL_OFF", 1.0
    return "SHOW", "NO_MEMORY_EFFECT", 1.0


def record_event(customer_id: str, product_code: str, event: Event, on: date, store: MemoryStore | None = None) -> None:
    (store or default_store()).record(customer_id, product_code, event, on)


def apply_memory(rows: list[dict], today: date, store: MemoryStore | None = None) -> list[dict]:
    """Return copies of one customer's rows, ordered for display, each with row["memory"].
    Hidden/cooled-off rows are kept (for RMs) after the shown ones. Inputs are not mutated."""
    if not rows:
        return []
    store = store or default_store()
    history = store.history(rows[0]["customer_id"])
    out = []
    for row in rows:
        state = _state(history, row["product_code"])
        action, reason, mult = decide(row, state, today)
        base = (row.get("ranking_score") or 0.0) if row.get("eligible_to_show") else 0.0
        new = copy.deepcopy(row)
        new["memory"] = {"action": action, "reason": reason, "base_score": base, "multiplier": mult,
                         "display_score": round(base * mult, 8), "state": state}
        out.append(new)
    out.sort(key=lambda r: (r["memory"]["multiplier"] == 0, -r["memory"]["display_score"], r["product_code"]))
    shown = 0
    for r in out:
        if r["memory"]["multiplier"] > 0:
            shown += 1
            r["memory"]["display_rank"] = shown
        else:
            r["memory"]["display_rank"] = None
    return out
