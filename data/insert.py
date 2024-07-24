import sqlite3
import random

# Connect to the SQLite database
# The database file 'linear.db' must already exist
conn = sqlite3.connect("linear.db")
cursor = conn.cursor()

# Generate 1000 random integers between 0 and 1000
random_numbers = [random.randint(0, 1000) for _ in range(1000)]
print(len(random_numbers))  # Print the number of generated random numbers

# Generate 'x' values and corresponding 'y' values
# 'y' values are calculated as 2 * 'x' + 3
x = [num for num in random_numbers]
y = [(2 * num) + 3 for num in random_numbers]

print(x, y)  # Print the generated 'x' and 'y' values

# Prepare data for insertion
data_to_insert = list(zip(x, y))

# Insert the data into the 'data' table
# Using parameterized queries to prevent SQL injection
cursor.executemany("INSERT INTO data (X, y) VALUES (?, ?)", data_to_insert)

# Fetch and print all data from the 'data' table to verify insertion
cursor.execute("SELECT * FROM data;")
data = cursor.fetchall()
for i in data:
    print(i)

# Commit the changes to the database
conn.commit()

# Close the connection to the database
conn.close()
