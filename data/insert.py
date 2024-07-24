import sqlite3
import random 

conn=sqlite3.connect("linear.db")
cursor=conn.cursor()

#Insert 
'''
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
'''
random_numbers=[random.randint(0,1000) for _ in range(1000) ]
print(len(random_numbers))
x=[num for num in random_numbers]
y=[(2*num)+3 for num in random_numbers]

print(x,y)
data_to_insert=list(zip(x,y))
cursor.executemany("insert into data(X,y) values (?,?)",data_to_insert)

cursor.execute("select * from data;")
data=cursor.fetchall()
for i in data:
    print(i)
    
conn.commit()
conn.close()