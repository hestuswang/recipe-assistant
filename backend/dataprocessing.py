import pandas as pd
import json

# === 1️⃣ 读取菜谱数据 ===
# 你可以把上面的数据保存为 Excel 或 CSV 文件，比如 recipe_data.xlsx
df = pd.read_excel("recipe_data.xlsx")  # 或 pd.read_csv("recipe_data.csv")


# === 2️⃣ 定义函数：根据每行生成 QA 对 ===
def generate_qa_from_row(row):
    title = row['title']
    qa_pairs = []

    # 解析字段（处理字符串格式的列表）
    def safe_json_parse(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except:
                return [x]
        return x

    ingredients = safe_json_parse(row['ingredients'])
    directions = safe_json_parse(row['directions'])
    ner = safe_json_parse(row['NER'])
    cookers = [c.strip() for c in row['cooker'].split(',') if c.strip()]

    # ---- Q1 菜名简介 ----
    qa_pairs.append({
        "question": f"什么是 {title}？",
        "answer": f"{title} 是一道菜谱，包含主要食材如 {', '.join(ner[:4])} 等，具体做法如下：{' '.join(directions[:2])} ..."
    })

    # ---- Q2 主要原料 ----
    qa_pairs.append({
        "question": f"{title} 需要哪些原料？",
        "answer": "、".join(ingredients)
    })

    # ---- Q3 做法步骤 ----
    qa_pairs.append({
        "question": f"如何制作 {title}？",
        "answer": "\n".join([f"{i + 1}. {d}" for i, d in enumerate(directions)])
    })

    # ---- Q4 关键食材（NER） ----
    qa_pairs.append({
        "question": f"{title} 包含哪些关键食材？",
        "answer": "、".join(ner)
    })

    # ---- Q5 使用的厨具 ----
    qa_pairs.append({
        "question": f"做 {title} 需要哪些厨具？",
        "answer": "、".join(cookers)
    })

    # ---- Q6 反向查找：根据部分食材找到菜 ----
    for ingredient in ner:
        qa_pairs.append({
            "question": f"有哪些菜谱包含 {ingredient}？",
            "answer": f"{title} 含有 {ingredient}，可以尝试这道菜。"
        })

    # ---- Q7 简易做法概述 ----
    qa_pairs.append({
        "question": f"我有 {ner[0]}、{ner[1]}，可以做什么？",
        "answer": f"你可以做 {title}。主要步骤包括：{directions[0]}"
    })

    return qa_pairs


# === 3️⃣ 生成所有 QA 对 ===
all_qa_pairs = []
for _, row in df.iterrows():
    all_qa_pairs.extend(generate_qa_from_row(row))

# === 4️⃣ 保存结果 ===
qa_df = pd.DataFrame(all_qa_pairs)
qa_df.to_json("recipe_qa_dataset.json", orient="records", force_ascii=False, indent=2)
qa_df.to_excel("recipe_qa_dataset.xlsx", index=False)

print("✅ 成功生成 QA 对，已保存为 recipe_qa_dataset.json 和 recipe_qa_dataset.xlsx")
