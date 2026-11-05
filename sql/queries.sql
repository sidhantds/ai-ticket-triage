-- =====================================================================
-- AI Ticket Triage - analytical queries
-- =====================================================================

-- 1. Ticket volume per predicted department
SELECT predicted_category AS category, COUNT(*) AS total_tickets
FROM predictions
GROUP BY predicted_category
ORDER BY total_tickets DESC;

-- 2. Priority breakdown with average confidence
SELECT predicted_priority AS priority,
       COUNT(*) AS tickets,
       ROUND(AVG(priority_confidence), 1) AS avg_confidence_pct
FROM predictions
GROUP BY predicted_priority
ORDER BY tickets DESC;

-- 3. Automation rate (auto-routed vs human review)
SELECT status,
       COUNT(*) AS tickets,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM predictions), 1) AS percent
FROM predictions
GROUP BY status;

-- 4. Supervisor queue: urgent tickets that need a human, least confident first
SELECT t.ticket_id, t.subject, p.predicted_category, p.category_confidence, p.sla_hours
FROM tickets t
JOIN predictions p ON t.ticket_id = p.ticket_id
WHERE p.predicted_priority = 'high'
  AND p.status = 'Needs Human Review'
ORDER BY p.category_confidence ASC
LIMIT 10;

-- 5. Model accuracy per category (tickets with known ground truth)
SELECT t.actual_category AS category,
       COUNT(*) AS total,
       SUM(CASE WHEN p.predicted_category = t.actual_category THEN 1 ELSE 0 END) AS correct,
       ROUND(100.0 * SUM(CASE WHEN p.predicted_category = t.actual_category THEN 1 ELSE 0 END)
             / COUNT(*), 1) AS accuracy_pct
FROM tickets t
JOIN predictions p ON t.ticket_id = p.ticket_id
WHERE t.actual_category IS NOT NULL
GROUP BY t.actual_category
ORDER BY accuracy_pct DESC;

-- 6. Workload matrix: category x priority
SELECT predicted_category AS category,
       SUM(CASE WHEN predicted_priority = 'high'   THEN 1 ELSE 0 END) AS high,
       SUM(CASE WHEN predicted_priority = 'medium' THEN 1 ELSE 0 END) AS medium,
       SUM(CASE WHEN predicted_priority = 'low'    THEN 1 ELSE 0 END) AS low,
       COUNT(*) AS total
FROM predictions
GROUP BY predicted_category
ORDER BY high DESC;

-- 7. Dangerous mistakes: truly high-priority tickets predicted as low
SELECT COUNT(*) AS high_predicted_as_low
FROM tickets t
JOIN predictions p ON t.ticket_id = p.ticket_id
WHERE t.actual_priority = 'high'
  AND p.predicted_priority = 'low';
