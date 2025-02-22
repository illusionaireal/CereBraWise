from openai import OpenAI
import os
import pandas as pd
import faiss
import numpy as np

client = OpenAI(
  api_key="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",
  base_url="https://integrate.api.nvidia.com/v1"
)
 
imageToId = "./data/dataText/imageToId.csv"
index_label_to_category = "./data/dataText/index_label_to_category.csv"

id_to_category = {}
imge_to_category = {}

import csv
 
print()

with open(index_label_to_category, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # 遍历每一行数据
    for row in reader:
        # 处理每一行数据
        name = row[1][row[1].find("Category:") + 9 :]
        id_to_category[row[0]] = name

# 打开CSV文件
with open(imageToId, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # 遍历每一行数据
    for row in reader:
        # 处理每一行数据
        imge_to_category[row[0]] = id_to_category[row[1]]
print(imge_to_category)


pathImg = "./data/dataImg"
ps = os.listdir(pathImg)
vectorMap = {}
vectorList = []
index = 0
for p in ps:
    path2file=pathImg+'/'+p
    print(path2file)
    response = client.embeddings.create(
        input=[ "tourist attraction",
        path2file],
        model="nvidia/nvclip",
        encoding_format="float"
    )
    # vectorMap[index] = response.data[1].embedding
    vectorList.append(response.data[1].embedding)
    index = index + 1
    print(response.data[1].embedding)
vector_np = np.array(vectorList)
shape = vector_np.shape[1]
vector_faiss = faiss.IndexFlatL2(shape)
vector_faiss.add(vector_np)
print('完成'+str(vector_faiss.ntotal)+'个向量的构建')
# for i in vectorMap:
#         print(vectorMap[i])