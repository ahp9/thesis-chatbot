import sqlite3
import pandas as pd

conn = sqlite3.connect("chainlit.db")

query = """
SELECT
  t.id AS thread_id,
  t."createdAt" AS thread_created_at,
  t.name AS thread_name,
  s.id AS step_id,
  s."createdAt" AS step_created_at,
  s.type,
  s.name,
  s.input,
  s.output
FROM threads t
LEFT JOIN steps s ON s."threadId" = t.id
ORDER BY t."createdAt", s."createdAt";
"""

df = pd.read_sql_query(query, conn)
df.to_csv("chainlit_history.csv", index=False)
df.to_json("chainlit_history.json", orient="records", indent=2)

conn.close()
