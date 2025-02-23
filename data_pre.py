from openai import OpenAI
import faiss
import numpy as np
import pickle

client = OpenAI(
    api_key="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",
    base_url="https://integrate.api.nvidia.com/v1"
)

imageToId = "./data/dataText/imageToId.csv"
index_label_to_category = "./data/dataText/index_label_to_category.csv"

id_to_category = {}
imge_to_category = {}

import csv

with open(index_label_to_category, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # 遍历每一行数据
    for row in reader:
        # 处理每一行数据
        name = row[1][row[1].find("Category:") + 9:]
        id_to_category[row[0]] = name

# 打开CSV文件
with open(imageToId, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # 遍历每一行数据
    for row in reader:
        # 处理每一行数据
        imge_to_category[row[0]] = id_to_category[row[1]]

with open('./vector.pkl', 'rb') as file:
    vectorList = pickle.load(file)

# 打印读取的List对象
vector_np = np.array(vectorList)
shape = vector_np.shape[1]
vector_faiss = faiss.IndexFlatL2(shape)
vector_faiss.add(vector_np)
print('完成' + str(vector_faiss.ntotal) + '个向量的构建')
# for i in vectorMap:
#         print(vectorMap[i])
