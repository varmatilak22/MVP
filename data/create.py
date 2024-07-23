import sqlite3

conn=sqlite3.connect('linear.db')
cursor=conn.cursor()

#Create table
cursor.execute("""
create table data(
X int,
y int)
""")

conn.commit()
conn.close()