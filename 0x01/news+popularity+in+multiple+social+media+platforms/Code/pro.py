import sqlite3
import csv

# 定义数据库文件名
db_file = 'example.db'

# 定义CSV文件名
csv_file = 'GooglePlus_Obama.csv'

# 定义表名
table_name = 'google_plus_data'

# 定义创建表的SQL语句，这里需要根据CSV的实际结构来修改
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    IDLink INTEGER PRIMARY KEY,
    TS1 INTEGER,
    TS2 INTEGER,
    TS3 INTEGER,
    -- ... 其他列 ...
    TS143 INTEGER,
    TS144 INTEGER
);
"""

# 建立数据库连接
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# 创建表
cursor.execute(create_table_sql)

# 插入CSV数据到数据库
def insert_data(row):
    columns = ','.join([col for col in row.keys()])
    values = ','.join(['?' for _ in row])
    sql = f'INSERT INTO {table_name} ({columns}) VALUES ({values})'
    cursor.executemany(sql, [tuple(row.values())])

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        insert_data(row)

# 提交事务
conn.commit()

# 关闭数据库连接
cursor.close()
conn.close()

print(f'Data from {csv_file} has been inserted into {db_file}.')