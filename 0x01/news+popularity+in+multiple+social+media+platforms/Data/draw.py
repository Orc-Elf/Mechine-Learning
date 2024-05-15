import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV数据
# 假设您的CSV文件名为"GooglePlus_Obama.csv"，请根据实际情况修改文件路径
df = pd.read_csv('GooglePlus_Obama.csv')

# 选择一个ID的数据进行绘制，这里以ID 61874为例
# 您可以更改IDLink的值来选择不同的ID
id_link = 61874
id_data = df[df['IDLink'] == id_link]

# 绘制趋势图
# 由于数据可能非常多，这里只绘制前几个时间序列作为示例
plt.figure(figsize=(14, 7))

# 假设我们只绘制前10个时间序列
for i in range(10):
    plt.plot(id_data.columns[1 + i::10], np.ravel(id_data.iloc[:, 1 + i::10].values), label=f'TS{10*i+1}')

# 旋转x轴标签，以便它们更易读
plt.xticks(rotation=45)

# 显示图例
plt.legend()

# 添加标题和轴标签
plt.title(f'Trend for ID {id_link}')
plt.xlabel('Time Series')
plt.ylabel('Value')

# 显示图表
plt.tight_layout()  # 调整布局以适应标签
plt.show()