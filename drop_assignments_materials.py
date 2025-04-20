import sqlite3

# Use your actual database filename here
conn = sqlite3.connect('backend/test.db')  # example: 'db.sqlite3', or whatever your SQLite file is called

cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS assignment_materials;")
conn.commit()
conn.close()

print("Table 'assignment_materials' dropped (if it existed).")