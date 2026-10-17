"""Evaluation set: 50 text-to-SQL questions with gold SQL + root-cause cases with ground truth.

Gold SQL deliberately mixes base tables and views so it does not share code with the agent's SQL.
Difficulty: easy = one aggregate; medium = group-by / filter / top-N; hard = ratios, CTEs, multi-step.
"""

Q = [
    # ---- easy ----
    ("easy", "What was the total revenue in 2024?", "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE year = 2024"),
    ("easy", "How many orders were placed in 2023?", "SELECT COUNT(*) FROM orders WHERE EXTRACT(year FROM order_date) = 2023"),
    ("easy", "What is the average order value overall?", "SELECT ROUND(SUM(revenue)/COUNT(DISTINCT order_id),2) FROM v_sales"),
    ("easy", "How many customers are registered?", "SELECT COUNT(*) FROM customers"),
    ("easy", "What is the overall cancellation rate?", "SELECT ROUND(AVG(CASE WHEN status='cancelled' THEN 1.0 ELSE 0 END),4) FROM orders"),
    ("easy", "What is the average delivery time in days?", "SELECT ROUND(AVG(delivery_days),2) FROM orders"),
    ("easy", "What is the average customer rating?", "SELECT ROUND(AVG(rating),2) FROM reviews"),
    ("easy", "How many units were sold in total?", "SELECT SUM(quantity) FROM v_sales"),
    ("easy", "What was total revenue in Q3 2024?", "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE year = 2024 AND quarter = 'Q3'"),
    ("easy", "How many orders were cancelled?", "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'"),
    # ---- medium: group-by ----
    ("medium", "Revenue by region", "SELECT region, ROUND(SUM(revenue),2) FROM v_sales GROUP BY region"),
    ("medium", "Revenue by category in 2024", "SELECT category, ROUND(SUM(revenue),2) FROM v_sales WHERE year = 2024 GROUP BY category"),
    ("medium", "Number of orders by region in 2024", "SELECT c.region, COUNT(*) FROM orders o JOIN customers c USING (customer_id) WHERE EXTRACT(year FROM o.order_date) = 2024 GROUP BY c.region"),
    ("medium", "Average delivery time by region", "SELECT c.region, ROUND(AVG(o.delivery_days),2) FROM orders o JOIN customers c USING (customer_id) GROUP BY c.region"),
    ("medium", "Monthly revenue for 2024", "SELECT month_start, ROUND(SUM(revenue),2) FROM v_sales WHERE year = 2024 GROUP BY month_start"),
    ("medium", "Revenue by quarter", "SELECT year, quarter, ROUND(SUM(revenue),2) FROM v_sales GROUP BY year, quarter"),
    ("medium", "Number of orders by status", "SELECT status, COUNT(*) FROM orders GROUP BY status"),
    ("medium", "Average rating by region", "SELECT c.region, ROUND(AVG(r.rating),2) FROM reviews r JOIN orders o USING (order_id) JOIN customers c USING (customer_id) GROUP BY c.region"),
    ("medium", "Revenue by customer type in 2024", "SELECT customer_type, ROUND(SUM(revenue),2) FROM v_sales WHERE year = 2024 GROUP BY customer_type"),
    ("medium", "Cancellation rate by region", "SELECT c.region, ROUND(AVG(CASE WHEN o.status='cancelled' THEN 1.0 ELSE 0 END),4) FROM orders o JOIN customers c USING (customer_id) GROUP BY c.region"),
    ("medium", "Units sold by category", "SELECT category, SUM(quantity) FROM v_sales GROUP BY category"),
    ("medium", "Average order value by region in 2024", "SELECT region, ROUND(SUM(revenue)/COUNT(DISTINCT order_id),2) FROM v_sales WHERE year = 2024 GROUP BY region"),
    ("medium", "How many customers placed orders in each year?", "SELECT EXTRACT(year FROM order_date), COUNT(DISTINCT customer_id) FROM orders GROUP BY 1"),
    ("medium", "Yearly revenue", "SELECT year, ROUND(SUM(revenue),2) FROM v_sales GROUP BY year"),
    # ---- medium: top-N ----
    ("medium", "Which region had the highest revenue in 2024?", "SELECT region FROM v_sales WHERE year = 2024 GROUP BY region ORDER BY SUM(revenue) DESC LIMIT 1"),
    ("medium", "Top 3 categories by revenue", "SELECT category FROM v_sales GROUP BY category ORDER BY SUM(revenue) DESC LIMIT 3"),
    ("medium", "Which month had the lowest revenue in 2023?", "SELECT month_start FROM v_sales WHERE year = 2023 GROUP BY month_start ORDER BY SUM(revenue) ASC LIMIT 1"),
    ("medium", "Top 5 months by number of orders", "SELECT month_start FROM orders o JOIN v_orders USING (order_id) GROUP BY month_start ORDER BY COUNT(*) DESC LIMIT 5"),
    ("medium", "Which category sold the most units in 2024?", "SELECT category FROM v_sales WHERE year = 2024 GROUP BY category ORDER BY SUM(quantity) DESC LIMIT 1"),
    ("medium", "Which region has the slowest average delivery?", "SELECT c.region FROM orders o JOIN customers c USING (customer_id) GROUP BY c.region ORDER BY AVG(o.delivery_days) DESC LIMIT 1"),
    ("medium", "Which quarter had the highest revenue?", "SELECT year, quarter FROM v_sales GROUP BY year, quarter ORDER BY SUM(revenue) DESC LIMIT 1"),
    ("medium", "Best selling product category in Q4 2023", "SELECT category FROM v_sales WHERE year = 2023 AND quarter = 'Q4' GROUP BY category ORDER BY SUM(revenue) DESC LIMIT 1"),
    # ---- medium: filters ----
    ("medium", "Revenue from the West region in 2024", "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE region = 'West' AND year = 2024"),
    ("medium", "How many orders did repeat customers place in 2024?", "SELECT COUNT(*) FROM v_orders WHERE customer_type = 'repeat' AND year = 2024"),
    ("medium", "Electronics revenue in Q2 2024", "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE category = 'Electronics' AND year = 2024 AND quarter = 'Q2'"),
    ("medium", "Average delivery time in the West region during Q3 2024", "SELECT ROUND(AVG(delivery_days),2) FROM v_orders WHERE region = 'West' AND year = 2024 AND quarter = 'Q3'"),
    ("medium", "Number of cancelled orders in the South region in 2023", "SELECT COUNT(*) FROM v_orders WHERE status = 'cancelled' AND region = 'South' AND year = 2023"),
    ("medium", "Revenue from new customers in 2023", "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE customer_type = 'new' AND year = 2023"),
    ("medium", "Average rating for orders in July 2024", "SELECT ROUND(AVG(rating),2) FROM v_orders WHERE year = 2024 AND month = 7"),
    ("medium", "How many orders were placed in December 2023?", "SELECT COUNT(*) FROM orders WHERE order_date >= DATE '2023-12-01' AND order_date < DATE '2024-01-01'"),
    # ---- hard ----
    ("hard", "What percentage of 2024 revenue came from repeat customers?", "SELECT ROUND(100.0 * SUM(CASE WHEN customer_type='repeat' THEN revenue END) / SUM(revenue), 2) FROM v_sales WHERE year = 2024"),
    ("hard", "Which region had the biggest revenue decline from Q2 2024 to Q3 2024?",
     "WITH r AS (SELECT region, SUM(CASE WHEN quarter='Q3' THEN revenue ELSE 0 END) - SUM(CASE WHEN quarter='Q2' THEN revenue ELSE 0 END) AS d FROM v_sales WHERE year = 2024 GROUP BY region) SELECT region FROM r ORDER BY d ASC LIMIT 1"),
    ("hard", "How many customers ordered in both 2023 and 2024?",
     "SELECT COUNT(*) FROM (SELECT customer_id FROM orders GROUP BY customer_id HAVING MIN(EXTRACT(year FROM order_date)) = 2023 AND MAX(EXTRACT(year FROM order_date)) = 2024 AND COUNT(DISTINCT EXTRACT(year FROM order_date)) = 2)"),
    ("hard", "What is the average number of orders per customer who ordered?", "SELECT ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT customer_id), 2) FROM orders"),
    ("hard", "Revenue growth from 2023 to 2024 in percent",
     "SELECT ROUND(100.0 * (SUM(CASE WHEN year=2024 THEN revenue END) / SUM(CASE WHEN year=2023 THEN revenue END) - 1), 2) FROM v_sales"),
    ("hard", "Which category has the highest average unit price?", "SELECT category FROM v_sales GROUP BY category ORDER BY AVG(unit_price) DESC LIMIT 1"),
    ("hard", "Revenue by region and customer type for Q3 2024", "SELECT region, customer_type, ROUND(SUM(revenue),2) FROM v_sales WHERE year = 2024 AND quarter = 'Q3' GROUP BY region, customer_type"),
    ("hard", "In which month of 2024 was the average delivery time the longest?", "SELECT month FROM v_orders WHERE year = 2024 GROUP BY month ORDER BY AVG(delivery_days) DESC LIMIT 1"),
    ("hard", "How many orders had a 5 star rating?", "SELECT COUNT(*) FROM reviews WHERE rating = 5"),
    ("hard", "What share of orders were cancelled in the West region in Q3 2024?", "SELECT ROUND(AVG(CASE WHEN status='cancelled' THEN 1.0 ELSE 0 END),4) FROM v_orders WHERE region = 'West' AND year = 2024 AND quarter = 'Q3'"),
]

QUESTIONS = [{"id": f"q{i + 1:02d}", "difficulty": d, "question": q, "gold_sql": s} for i, (d, q, s) in enumerate(Q)]

# Root-cause cases. `expect` keys are checked against the Investigation facts.
RCA_CASES = [
    {"question": "Why did revenue drop in Q3?", "expect": {"window": "Q3 2024", "filters": {"region": "West", "customer_type": "repeat"}, "mechanism": "delivery_days"}},
    {"question": "What caused the decline in sales in the third quarter of 2024?", "expect": {"window": "Q3 2024", "filters": {"region": "West", "customer_type": "repeat"}, "mechanism": "delivery_days"}},
    {"question": "Revenue fell in Q3 2024 - explain why", "expect": {"window": "Q3 2024", "filters": {"region": "West", "customer_type": "repeat"}, "mechanism": "delivery_days"}},
    {"question": "Why did sales decrease in Q3 2024?", "expect": {"window": "Q3 2024", "filters": {"region": "West", "customer_type": "repeat"}, "mechanism": "delivery_days"}},
    {"question": "Why did revenue drop recently?", "expect": {"window": "Q3 2024", "filters": {"region": "West", "customer_type": "repeat"}, "mechanism": "delivery_days"}},
    {"question": "Why did orders drop in Q3 2024 compared to last year?", "expect": {"window": "Q3 2024", "filters": {"region": "West"}, "mechanism": "delivery_days"}},
    # negative controls: the agent must NOT invent a cause
    {"question": "Why did revenue drop in Q4 2024?", "expect": {"no_change_in_direction": True}},
    {"question": "Why did revenue fall in Q1 2024?", "expect": {"broad_based": True}},
]
