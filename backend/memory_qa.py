"""
运行后自动完成：
1. 建库 recipe_memory_crud.db
2. 导 recipes_data.csv → recipes 表
3. 导 recipe_qa_dataset.xlsx → qa_pairs 表 + embedding
4. 内存加载向量矩阵
5. 打印 3 条测试检索最相关结果
"""

import json
import sqlite3
import pathlib
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ---------- 路径 ----------
DB_FILE     = pathlib.Path("recipe_memory_crud.db")
CSV_FILE    = pathlib.Path("recipes_data_new_test.csv")          # 原始菜谱
QA_FILE     = pathlib.Path("recipe_qa_dataset_test.xlsx")    # 生成的 QA 对
EMB_MODEL   = SentenceTransformer("/Users/hestuswang/Desktop/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")

# =============================================================================
#  数据库建表
# =============================================================================
def _init_db(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS recipes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            ingredients TEXT,
            directions TEXT,
            ner TEXT,
            cooker TEXT,
            embedding BLOB
        );
        CREATE TABLE IF NOT EXISTS qa_pairs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER,
            question TEXT,
            answer TEXT,
            embedding BLOB
        );
        CREATE TABLE IF NOT EXISTS user_profile(
            user_id TEXT PRIMARY KEY,
            forbidden_ingredients TEXT,
            taste_level TEXT,
            unavailable_utensils TEXT,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS session_cache(
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            raw_ingredients TEXT,
            temporary_req TEXT,
            rejected_recipes TEXT,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            recipe_id INTEGER,
            reason TEXT,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()

# =============================================================================
#  1. 导菜谱（只导不入向量）
# =============================================================================
def _load_recipes(conn: sqlite3.Connection):
    if conn.execute("SELECT 1 FROM recipes LIMIT 1").fetchone():
        return
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"请把 {CSV_FILE} 放在当前目录！")
    df = pd.read_csv(CSV_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    def _safe_list(cell):
        if pd.isna(cell):
            return []
        try:
            return json.loads(cell)
        except Exception:
            return [x.strip() for x in str(cell).split(",") if x.strip()]

    records = [
        (row["title"],
         json.dumps(_safe_list(row["ingredients"])),
         json.dumps(_safe_list(row["directions"])),
         json.dumps(_safe_list(row["ner"])),
         json.dumps(_safe_list(row["cooker"])))
        for _, row in df.iterrows()
    ]
    conn.executemany(
        "INSERT INTO recipes(name,ingredients,directions,ner,cooker) VALUES (?,?,?,?,?)",
        records,
    )
    conn.commit()

# =============================================================================
#  2. 导 QA 对 + 立即 embedding
# =============================================================================
# def _load_qa_pairs(conn: sqlite3.Connection):
#     if conn.execute("SELECT 1 FROM qa_pairs LIMIT 1").fetchone():
#         return
#     if not QA_FILE.exists():
#         raise FileNotFoundError(f"请把 {QA_FILE} 放在当前目录！")
#     df = pd.read_excel(QA_FILE)
#     for _, row in df.iterrows():
#         q, a = row["question"], row["answer"]
#         vec = EMB_MODEL.encode(q + " " + a, normalize_embeddings=True).astype(np.float32)
#         conn.execute(
#             "INSERT INTO qa_pairs(recipe_id,question,answer,embedding) VALUES (?,?,?,?)",
#             (None, q, a, vec.tobytes())
#         )
#     conn.commit()

def _load_qa_pairs(conn: sqlite3.Connection):
    if conn.execute("SELECT 1 FROM qa_pairs LIMIT 1").fetchone():
        return
    if not QA_FILE.exists():
        raise FileNotFoundError(f"请把 {QA_FILE} 放在当前目录！")
    df = pd.read_excel(QA_FILE)
    # 统一转字符串防错
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"]   = df["answer"].astype(str).str.strip()

    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        q, a = row["question"], row["answer"]
        vec = EMB_MODEL.encode(q + " " + a, normalize_embeddings=True).astype(np.float32)
        conn.execute(
            "INSERT INTO qa_pairs(recipe_id,question,answer,embedding) VALUES (?,?,?,?)",
            (None, q, a, vec.tobytes())
        )
        # 每 1000 条或最后一条刷新进度
        if idx % 1000 == 0 or idx == total:
            print(f"\r>>> QA embedding 进度: {idx}/{total} ({idx*100//total:>3}%)", end="")
    print()
    conn.commit()

# =============================================================================
#  3. 把 QA 向量加载到内存
# =============================================================================
def _load_qa_torch(conn: sqlite3.Connection):
    if hasattr(RecipeMemory, "qa_emb_mat"):
        return

    rows = list(conn.execute("SELECT id,embedding FROM qa_pairs WHERE embedding IS NOT NULL"))
    if not rows:
        raise RuntimeError("qa_pairs 没有任何向量可加载！")

    total = len(rows)
    ids, embs = [], []
    for idx, (rid, vec_bytes) in enumerate(rows, 1):
        ids.append(rid)
        embs.append(np.frombuffer(vec_bytes, dtype=np.float32))
        # 每 1000 条或最后一条刷新
        if idx % 1000 == 0 or idx == total:
            print(f"\r>>> 向量加载进度: {idx}/{total} ({idx*100//total:>3}%)", end="")
    print()

    RecipeMemory.qa_ids   = np.array(ids, dtype=np.int32)
    RecipeMemory.qa_emb_mat = torch.from_numpy(np.vstack(embs))
    if torch.cuda.is_available():
        RecipeMemory.qa_emb_mat = RecipeMemory.qa_emb_mat.cuda()

# =============================================================================
#  RecipeMemory 类
# =============================================================================
class RecipeMemory:
    qa_ids: np.ndarray
    qa_emb_mat: torch.Tensor

    def __init__(self, db: pathlib.Path = DB_FILE):
        self.db = db
        need_create = not db.exists()
        self._conn = sqlite3.connect(db, check_same_thread=False)
        _init_db(self._conn)
        if need_create:
            print(">>> 首次运行，开始建库并导入数据 ...")
            _load_recipes(self._conn)
            print(">>> 菜谱数据导入完成，开始导入QA并embedding ...")
            _load_qa_pairs(self._conn)
        print(">>> 数据导入完成，加载向量矩阵 ...")
        _load_qa_torch(self._conn)
        print(">>> RecipeMemory 初始化完成！")

    # ---------------- 对外检索接口 ----------------
    def search_qa(self, query: str, k: int = 5) -> List[int]:
        """返回最相关的 k 个 qa_pairs.id（按相似度降序）"""
        qvec = EMB_MODEL.encode(query, normalize_embeddings=True).astype(np.float32)
        qtensor = torch.from_numpy(qvec).unsqueeze(0)
        if self.qa_emb_mat.is_cuda:
            qtensor = qtensor.cuda()
        scores = torch.matmul(self.qa_emb_mat, qtensor.T).squeeze()
        # 取前 k 个下标
        topk_vals, topk_idx = torch.topk(scores, k=min(k, scores.size(0)))
        return self.qa_ids[topk_idx.cpu().numpy()].tolist()

    # ---------------- QA 对 CRUD ----------------
    def add_qa(self, question: str, answer: str, recipe_id: Optional[int] = None) -> int:
        # 新增一条 QA 对并立即生成 embedding，返回新插入行的 id
        vec = EMB_MODEL.encode(question + " " + answer, normalize_embeddings=True).astype(np.float32)
        cur = self._conn.execute(
            "INSERT INTO qa_pairs(recipe_id, question, answer, embedding) VALUES (?, ?, ?, ?)",
            (recipe_id, question, answer, vec.tobytes())
        )
        self._conn.commit()
        return cur.lastrowid

    def get_qa(self, qa_id: int) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT id, recipe_id, question, answer FROM qa_pairs WHERE id=?",
            (qa_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "recipe_id": row[1],
            "question": row[2],
            "answer": row[3]
        }

    def update_qa(self, qa_id: int, *, question: Optional[str] = None,
                  answer: Optional[str] = None, recipe_id: Optional[int] = None) -> None:
        # 增量更新 QA 对；若修改 question/answer 则同步重新 embedding
        old = self.get_qa(qa_id)
        if not old:
            raise KeyError(f"qa_id={qa_id} 不存在")

        new_q = question if question is not None else old["question"]
        new_a = answer if answer is not None else old["answer"]
        new_rid = recipe_id if recipe_id is not None else old["recipe_id"]

        # 只要改了 question 或 answer 就重算向量
        if (question is not None) or (answer is not None):
            vec = EMB_MODEL.encode(new_q + " " + new_a, normalize_embeddings=True).astype(np.float32)
            self._conn.execute(
                "UPDATE qa_pairs SET question=?, answer=?, embedding=?, recipe_id=? WHERE id=?",
                (new_q, new_a, vec.tobytes(), new_rid, qa_id)
            )
        else:
            self._conn.execute(
                "UPDATE qa_pairs SET recipe_id=? WHERE id=?",
                (new_rid, qa_id)
            )
        self._conn.commit()

    def del_qa(self, qa_id: int) -> None:
        self._conn.execute("DELETE FROM qa_pairs WHERE id=?", (qa_id,))
        self._conn.commit()

    # ---------------- 用户/会话/反馈 CRUD（保持不变） ----------------
    def add_user_profile(self, user_id: str,
                         forbidden: List[str],
                         taste: Dict[str, str],
                         unavailable_utensils: List[str]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO user_profile
               (user_id,forbidden_ingredients,taste_level,unavailable_utensils)
               VALUES (?,?,?,?)""",
            (user_id, json.dumps(forbidden), json.dumps(taste), json.dumps(unavailable_utensils)),
        )
        self._conn.commit()

    def get_user_profile(self, user_id: str) -> Dict:
        row = self._conn.execute(
            "SELECT forbidden_ingredients,taste_level,unavailable_utensils FROM user_profile WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if not row:
            return {}
        return {
            "forbidden_ingredients": json.loads(row[0]) if row[0] else [],
            "taste_level": json.loads(row[1]) if row[1] else {},
            "unavailable_utensils": json.loads(row[2]) if row[2] else [],
        }

    def update_user_profile(self, user_id: str, delta: Dict) -> None:
        old = self.get_user_profile(user_id)
        old.update(delta)
        self.add_user_profile(user_id, **old)

    def del_user_profile(self, user_id: str) -> None:
        self._conn.execute("DELETE FROM user_profile WHERE user_id=?", (user_id,))
        self._conn.commit()

    # ---------- session ----------
    def add_session(self, session_id: str, user_id: str,
                    raw_ing: List[str], temp_req: Dict, rejected: List[int]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO session_cache
               (session_id,user_id,raw_ingredients,temporary_req,rejected_recipes)
               VALUES (?,?,?,?,?)""",
            (session_id, user_id, json.dumps(raw_ing), json.dumps(temp_req), json.dumps(rejected)),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Dict:
        row = self._conn.execute(
            "SELECT user_id,raw_ingredients,temporary_req,rejected_recipes FROM session_cache WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if not row:
            return {}
        return {
            "user_id": row[0],
            "raw_ingredients": json.loads(row[1]) if row[1] else [],
            "temporary_req": json.loads(row[2]) if row[2] else {},
            "rejected_recipes": json.loads(row[3]) if row[3] else [],
        }

    def update_session(self, session_id: str, delta: Dict) -> None:
        old = self.get_session(session_id)
        old.update(delta)
        self.add_session(session_id, **old)

    def del_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM session_cache WHERE session_id=?", (session_id,))
        self._conn.commit()

    # ---------- feedback ----------
    def add_feedback(self, user_id: str, recipe_id: int, reason: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO feedback(user_id,recipe_id,reason) VALUES (?,?,?)",
            (user_id, recipe_id, reason),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_feedback(self, user_id: Optional[str] = None, recipe_id: Optional[int] = None) -> List[Dict]:
        sql, params = "SELECT id,user_id,recipe_id,reason,create_time FROM feedback WHERE 1=1", []
        if user_id:
            sql += " AND user_id=?"
            params.append(user_id)
        if recipe_id:
            sql += " AND recipe_id=?"
            params.append(recipe_id)
        sql += " ORDER BY create_time DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {"id": r[0], "user_id": r[1], "recipe_id": r[2], "reason": r[3], "create_time": r[4]}
            for r in rows
        ]

    def del_feedback(self, fid: int) -> None:
        self._conn.execute("DELETE FROM feedback WHERE id=?", (fid,))
        self._conn.commit()

# =============================================================================
#  一键测试：首次运行会打印 3 条检索结果
# =============================================================================
if __name__ == "__main__":
    mem = RecipeMemory()
    query = "What is No-Bake Nut Cookies?"
    top_k = 3
    ids = mem.search_qa(query, k=top_k)

    print(f">>> 问题：{query}")
    print(f">>> 最相关的 {top_k} 条 QA 对：\n")
    for rank, qid in enumerate(ids, 1):
        row = mem.get_qa(qid)
        if row:
            print(f"{rank}. [Q] {row['question']}\n   [A] {row['answer']}\n")