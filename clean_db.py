import sqlite3

db_path = r"D:\Git_Repos\ai-doc-assistant\data\history.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Delete all rows from each table
for table in tables:
    cursor.execute(f"DELETE FROM {table[0]};")

conn.commit()
conn.close()
print("history.db cleared.")