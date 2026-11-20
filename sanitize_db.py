"""
One-time script to sanitize contaminated conversation_history rows in haimiya.db.
Removes assistant rows that contain leaked internal reasoning.
Run once: python sanitize_db.py
"""
import pathlib
import re
import sqlite3

DB_PATH = pathlib.Path("data/haimiya.db")

# Patterns that indicate internal reasoning leaked into stored assistant messages
REASONING_PATTERNS = [
    re.compile(r"okay,?\s+let'?s\s+see", re.IGNORECASE),
    re.compile(r"the\s+user\s+(is|asked|wants|said)\b", re.IGNORECASE),
    re.compile(r"first,?\s+i\s+need\s+to\s+(respond|answer|reply|think|make|speak)\b", re.IGNORECASE),
    re.compile(r"wait,?\s+the\s+response\s+should\s+be\b", re.IGNORECASE),
    re.compile(r"i\s+should\s+(respond|answer|reply|act|speak|use|avoid|pick)\b", re.IGNORECASE),
    re.compile(r"i\s+need\s+to\s+(respond|answer|check|make|ensure)\b", re.IGNORECASE),
    re.compile(r"possible\s+response\s*:", re.IGNORECASE),
    re.compile(r"let\s+me\s+think\b", re.IGNORECASE),
    re.compile(r"we\s+are\s+in\s+a\s+conversation\s+where\b", re.IGNORECASE),
    re.compile(r"since\s+the\s+user\s+is\b", re.IGNORECASE),
]


def is_contaminated(content: str) -> bool:
    for pat in REASONING_PATTERNS:
        if pat.search(content):
            return True
    return False


def sanitize():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, content FROM conversation_history WHERE role = 'assistant'"
    ).fetchall()

    contaminated_ids = [row[0] for row in rows if is_contaminated(row[1])]

    if not contaminated_ids:
        print("No contaminated rows found. Database is clean.")
        conn.close()
        return

    print(f"Found {len(contaminated_ids)} contaminated assistant rows: {contaminated_ids}")

    # Delete contaminated assistant rows AND their corresponding user rows
    # (to avoid orphaned one-sided exchanges in history)
    placeholders = ",".join("?" * len(contaminated_ids))

    # Find partner user rows: they are the row immediately before each contaminated assistant row
    partner_ids = []
    for cid in contaminated_ids:
        partner = conn.execute(
            """
            SELECT id FROM conversation_history
            WHERE role = 'user' AND id < ?
            ORDER BY id DESC LIMIT 1
            """,
            (cid,),
        ).fetchone()
        if partner:
            partner_ids.append(partner[0])

    all_ids = list(set(contaminated_ids + partner_ids))
    all_placeholders = ",".join("?" * len(all_ids))
    deleted = conn.execute(
        f"DELETE FROM conversation_history WHERE id IN ({all_placeholders})",
        all_ids,
    ).rowcount
    conn.commit()
    conn.close()
    print(f"Deleted {deleted} rows (contaminated assistant + orphaned user rows).")
    print("Database sanitized successfully.")


if __name__ == "__main__":
    sanitize()
