import sqlite3

conn=sqlite3.connect("linear.db")
cursor=conn.cursor()

#Insert 

cursor.execute("""
insert into data values 
(1,2),
(2,4),
(3,6),
(4,8),
(5,10),
(6,12),
(7,14),
(8,16),
(9,18),
(10,20)
""")


cursor.execute("select * from data;")
data=cursor.fetchall()
for i in data:
    print(i)
    
conn.commit()
conn.close()