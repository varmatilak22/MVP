import sqlite3

# Connect to the SQLite database
# The database file 'linear.db' will be created if it does not exist
conn = sqlite3.connect('linear.db')
cursor = conn.cursor()

# Create a new table called 'data'
# The table has two columns: X and y, both of type integer
cursor.execute("""
CREATE TABLE data (
    X INT,  # Column for feature X
    y INT   # Column for target y
)
""")

# Commit the changes to the database
conn.commit()

# Close the connection to the database
conn.close()
