-- =====================================================================
-- AI Ticket Triage - SQLite schema
-- Facts (tickets) are kept separate from model opinions (predictions),
-- so a retrained model can add new predictions without touching tickets.
-- =====================================================================

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    subject          TEXT,
    body             TEXT NOT NULL,
    actual_category  TEXT,          -- ground truth (known for held-out test tickets only)
    actual_priority  TEXT,          -- ground truth (known for held-out test tickets only)
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- stored in UTC
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id            INTEGER NOT NULL,
    predicted_category   TEXT,
    category_confidence  REAL,      -- 0-100
    predicted_priority   TEXT,      -- high | medium | low
    priority_confidence  REAL,      -- 0-100
    sla_hours            INTEGER,   -- 1 | 8 | 24
    status               TEXT,      -- Auto-Routed | Needs Human Review
    key_words            TEXT,      -- comma-separated top TF-IDF terms
    model_version        TEXT,
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
);
