## 一、系统核心流程图
```mermaid
graph TD
    A[用户输入] --> B{多模态输入}
    B -->|图片| C[视觉特征提取]
    B -->|文本| D[语义编码]
    C --> E[多模态融合]
    D --> E
    E --> F[FAISS混合检索]
    F --> G[知识库上下文获取]
    G --> H[LLM行程生成]
    H --> I[Markdown渲染]
    I --> J[图文输出]
```

## 二、核心流程详解
### 阶段1：多模态输入处理

1. 输入捕获
- Gradio界面同时接收：

    - 图像输入：支持JPG/PNG格式（最大5MB）
    - 文本输入：200字以内自然语言描述
    - 示例输入组合：
```
inputs = {
    "image": "mountain.jpg", 
    "text": "计划3天徒步行程，偏好生态露营"
}
```
2. 并行编码
- 视觉通道：
```
# 使用NVIDIA NIM视觉服务
img_embed = nim.call_service(
    service="multimodal_encoder",
    params={"modality": "image"},
    inputs={"data": image_bytes}
)
```

- 文本通道：

```
    # 使用NVIDIA NIM文本服务
text_embed = nim.call_service(
    service="multimodal_encoder", 
    params={"modality": "text"},
    inputs={"data": input_text}
)
```
### 阶段2：联合检索
1. 特征融合

- 采用加权拼接策略：
```
fused_embed = np.concatenate([
    0.6 * img_embed,  # 视觉特征权重
    0.4 * text_embed  # 文本特征权重
])
```

- L2归一化处理保证检索稳定性
2. 混合检索

- FAISS索引配置：
```
index = faiss.IndexHNSWFlat(
    dimension=1024,  # 融合后维度
    M=32,            # HNSW层级参数
    metric=faiss.METRIC_INNER_PRODUCT
)
```

- 执行多模态检索：
```
D, I = index.search(fused_embed, k=3)  # 返回top3结果
```

###　阶段3：行程生成（关键路径耗时<3秒）
1. 上下文构建

- 动态提示工程：
```
prompt_template = """
[系统角色] 专业旅行规划师
[用户需求] {user_text}
[相关景点] {retrieved_pois}
[图片特征] {image_tags}

请生成包含以下要素的行程：
1. 每日时间轴（7:00-22:00） 
2. 交通衔接建议
3. 必体验项目（带emoji标注）
"""
```
- 结构化生成

  - 调用NIM对话服务：
```
response = nim.call_service(
    service="itinerary_llm",
    inputs={
        "prompt": constructed_prompt,
        "response_format": "markdown"
    },
    params={
        "temperature": 0.3,
        "max_tokens": 1500
    }
)
```

### 阶段4：结果渲染
1. 图文合成

- Markdown增强方案：
``` md
## 三日徒步行程
![露营点](data:image/jpeg;base64,{thumbnail})  
**Day1**  
✅ 07:00 山脚集合 → 🚌 乘坐接驳车  
🌳 09:00-12:00 雨林穿越（需携带：...）
```

2. 输出优化
```
自动生成缩略图：
def generate_thumbnail(img):
    return cv2.resize(img, (320, 240), 
        interpolation=cv2.INTER_AREA)
```


## 三、项目排期

```mermaid
gantt
    title 智能行程规划系统开发排期
    dateFormat  YYYY-MM-DD HH:mm
    axisFormat %d-%H

    section 蒋纹
    多模态输入验证       :a1, 2025-02-21 08:30, 14.5h
    混合检索 验证        :a2, 2025-02-21 08:30, 14.5h
    2 :a3, 2025-02-22 08:30, 14.5h
    3      :c3, 2025-02-23 08:30, 4.5h
    文档       :a4, 2025-02-23 13:00, 4h

    section 刘程萌
    数据集FAISS 预处理 验证 :b1, 2025-02-21  08:30, 14.5h
    混合检索 验证 :b2 ,2025-02-21  08:30, 14.5h
    2  :b3, 2025-02-22  08:30, 14.5h
    3    :b4, 2025-02-23  08:30, 4.5h
    文档       :a4, 2025-02-23 13:00, 4h

    section 曾驳 
    Jetson demo验证    :c1, 2025-02-21, 14.5h
    检索结果到输出的提示工程开发     :c2, 2025-02-21, 2d
    3      :c3, 2025-02-23 08:30, 4.5h
    文档       :a4, 2025-02-23 13:00, 4h
```

## 四：测试对话

[AI] 请问您想要什么样的旅游呢？
[用户] 我想去阳光明媚的地方 + 图片
[AI] 推荐景点：
1. 三亚亚龙湾：碧海银沙，热带风情...
2. 厦门鼓浪屿：文艺小岛，百年建筑...
3. 大理洱海：苍山映湖，白族文化...
请选择（输入编号或名称）：
[用户] 大理洱海
[AI] 推荐路线：
1. 环湖骑行路线：专业自行车道...
2. 文化探访路线：白族民居体验...
3. 摄影打卡路线：最佳取景点...
请选择：
[用户] 路线2
[AI] 显示Markdown格式的详细攻略


