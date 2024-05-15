import pandas as pd

# 加载CSV文件
df = pd.read_csv('GooglePlus_Obama.csv')

# 检查缺失数据
missing_data = df.isnull().sum()
print("Missing data per column:")
print(missing_data)

# 处理缺失数据，这里我们选择删除包含缺失数据的行
df = df.dropna()

# 检查数据类型
print("\nData types of each column:")
print(df.dtypes)

# 假设我们想将所有的数据转换为整数类型
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')  # 转换为数值类型，错误值将变为NaN
    df = df.dropna()  # 再次删除包含NaN的行

# 检查错误数据，例如，我们期望的数值范围是0到100
for column in df.columns:
    if 'TS' in column:
        # 移除不在期望范围内的数据
        df = df[df[column].between(0, 100, inclusive=True)]

# 数据格式统一和归一化
# 假设我们想将所有数据归一化到0到1之间
for column in df.columns:
    if 'TS' in column:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# 保存清洗后的数据到新的CSV文件
df.to_csv('cleaned_GooglePlus_Obama.csv', index=False)

print("\nData cleaning completed. Cleaned data saved to 'cleaned_GooglePlus_Obama.csv'.")