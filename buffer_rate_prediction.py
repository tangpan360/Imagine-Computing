# %%
# 导入数据处理包
import pandas as pd

# %%
# 导入数据
train_path = './dataset/training_dataset.csv'
test_path = './dataset/test_dataset.csv'
# 训练集的 dataframe
df_train = pd.read_csv(train_path)  
# 测试集的 dataframe
df_test = pd.read_csv(test_path)  
print("训练数据集：", df_train.shape,"\n测试数据集：", df_test.shape)

# %%
rowNum_train = df_train.shape[0]
rowNum_test = df_test.shape[0]
print(f"训练数据集有多少行数据：", format(rowNum_train, ','),
      "\n测试数据集有多少行数据：", format(rowNum_test, ','))

# %%
# 合并数据集，方便同时对两个数据集进行清洗
full = df_train.append(df_test, ignore_index=True)  
print("合并后的数据集：", full.shape)


# %%
# 查看数据前5行
full.head()  

# %%
# 查看数据的 8_999_998 至 9_000_003 行
full.loc[8_999_998:9_000_003, :]  

# %%
# 获取数据类型列的描述统计信息
full.describe()  

# %%
# 查看每一列的数据类型、和数据总数以及数据缺失情况
full.info(null_counts=True)  

# %%
# 数据集形状
full.shape  

# %% [markdown]
# 相关系数：***衡量相似度程度***，当他们的相关系数为1时，说明两个变量变化时的正向相似度最大，当相关系数为－1时，说明两个变量变化的反向相似度最大

# %%
# 斯皮尔曼相关系数矩阵
corrDf = full.corr(method='spearman')  
corrDf

# %%
"""
查看各个特征与卡顿率（buffer_rate ）的相关系数，
ascending=False 表示按降序排列
"""
corrDf['buffer_rate'].sort_values(ascending=False)

# %%
# 特征选择：根据各个特征与卡顿率（buffer_rate）的相关系数大小，选相关系数大于绝对值大于 0.05 的特征作为模型的输入
full_X = pd.concat([full['icmp_lossrate'],
                    full['icmp_rtt'],
                    full['avg_fbt_time'],
                    full['tcp_conntime'],
                    full['ratio_499_5xx'],
                    full['cpu_util'],
                    full['inner_network_droprate'],
                    full['synack1_ratio']], axis=1)

full_X.head()

# %%
sourceRow = 9_000_000
# 选取相关性较高的特征矩阵前 9_000_000 行作为输入特征矩阵
source_X = full_X.loc[0:sourceRow-1, :]  
# 选原训练和测试矩阵拼接的前 9_000_000 行 buffer_rate 作为输出标签
source_y = full.loc[0:sourceRow-1, 'buffer_rate']  

# 选取相关性较高的特征矩阵 9_000_000 至最后作为预测的输入特征矩阵
pred_X = full_X.loc[sourceRow:, :]  

# %%
print("原始训练集有多少行：", source_X.shape[0])
print("原始预测集有多少行：", pred_X.shape[0])

# %%
from sklearn.model_selection import train_test_split
# 建立模型用的训练数据集和测试集
train_X, test_X, train_y, test_y = train_test_split(source_X, source_y, train_size=0.8)

print ('原始数据集特征：',source_X.shape,
       '训练数据集特征：',train_X.shape ,
      '测试数据集特征：',test_X.shape)

print ('原始数据集标签：',source_y.shape,
       '训练数据集标签：',train_y.shape ,
      '测试数据集标签：',test_y.shape)

# %%
source_y.head()

# %%
# 导入算法
from sklearn.linear_model import LinearRegression
# 创建线性回归模型
model = LinearRegression()

# %%
# 训练模型
model.fit(train_X, train_y)

# %%
# 评估模型
model.score(test_X, test_y)

# %%



