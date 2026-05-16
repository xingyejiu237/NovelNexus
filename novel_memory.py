import os
import re
import glob
from typing import Optional

# 强制离线 + 国内镜像，避免 HuggingFace 连不上导致卡死
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config


_EMBEDDER_CACHE = {}
_CLIENT_CACHE = {}


class _ChromaEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str):
        if model_name not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
        self.model = _EMBEDDER_CACHE[model_name]

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, show_progress_bar=False).tolist()


class NovelMemory:

    def __init__(self, novel_name: Optional[str] = None, lazy: bool = False):
        self.novel_name = novel_name or config.ACTIVE_NOVEL
        self._collection = None
        self._embedding_fn = None
        if not lazy:
            self._ensure_collection()

    def _get_db_path(self) -> str:
        base = os.environ.get("NOVEL_MEMORY_DIR") or os.path.join(
            os.environ.get("TEMP", os.path.expanduser("~")),
            "novelnexus_memory"
        )
        return os.path.join(base, self.novel_name, "memory_db")

    def _get_embedding_fn(self):
        if self._embedding_fn is None:
            self._embedding_fn = _ChromaEmbedder(config.VECTOR_MEMORY_MODEL)
        return self._embedding_fn

    def _get_client(self):
        db_path = self._get_db_path()
        if db_path not in _CLIENT_CACHE:
            os.makedirs(db_path, exist_ok=True)
            try:
                _CLIENT_CACHE[db_path] = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False),
                )
            except Exception:
                return None
        return _CLIENT_CACHE[db_path]

    def _ensure_collection(self):
        if self._collection is not None:
            return
        client = self._get_client()
        if client is None:
            return
        ef = self._get_embedding_fn()
        self._collection = client.get_or_create_collection(
            name="chapters",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chapter(self, chapter_num: int, text: str):
        self._ensure_collection()
        if self._collection is None:
            return
        paragraphs = self._split_paragraphs(text, chapter_num)
        if not paragraphs:
            return
        existing_ids = set(self._collection.get(ids=[p["id"] for p in paragraphs])["ids"])
        new_paragraphs = [p for p in paragraphs if p["id"] not in existing_ids]
        if not new_paragraphs:
            return
        ids = [p["id"] for p in new_paragraphs]
        documents = [p["text"] for p in new_paragraphs]
        metadatas = [{"chapter": chapter_num, "paragraph_index": p["index"]} for p in new_paragraphs]
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query: str, k: int = None) -> list[dict]:
        self._ensure_collection()
        if self._collection is None:
            return []
        k = k or config.VECTOR_MEMORY_SEARCH_K
        count = self._collection.count()
        if count == 0:
            return []
        actual_k = min(k, count)
        results = self._collection.query(
            query_texts=[query],
            n_results=actual_k,
        )
        snippets = []
        max_ch = 1
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            ch = meta["chapter"]
            if ch > max_ch:
                max_ch = ch
            snippets.append({
                "chapter": ch,
                "paragraph_index": meta["paragraph_index"],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        boost = config.VECTOR_MEMORY_RECENCY_BOOST
        if boost > 0 and max_ch > 1:
            for s in snippets:
                if s["distance"] is not None:
                    s["distance"] = s["distance"] * (1.0 - boost * (s["chapter"] / max_ch))
            snippets.sort(key=lambda x: x["distance"] if x["distance"] is not None else 999)
        return snippets

    def search_keyword(self, keyword: str, k: int = 20) -> list[dict]:
        self._ensure_collection()
        if self._collection is None:
            return []
        results = self._collection.get(
            where_document={"$contains": keyword},
            limit=k,
        )
        snippets = []
        ids = results.get("ids", [])
        for i in range(len(ids)):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            doc = results["documents"][i] if results.get("documents") else ""
            snippets.append({
                "chapter": meta.get("chapter"),
                "paragraph_index": meta.get("paragraph_index"),
                "text": doc,
            })
        return snippets

    def recall_for_character(self, name: str, k: int = 5) -> list[dict]:
        return self.search(f"关于{name}的描写、外貌、行为、对话、动作", k=k)

    def recall_for_item(self, item_name: str, k: int = 5) -> list[dict]:
        return self.search(f"{item_name}的描述、外观、位置、使用、出现", k=k)

    def rebuild_index(self):
        self._collection = None
        db_path = self._get_db_path()
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
        self._ensure_collection()
        output_dir = config.get_novel_output_dir(self.novel_name)
        if not os.path.isdir(output_dir):
            return 0
        chapter_files = sorted(
            glob.glob(os.path.join(output_dir, "第*.md")),
            key=self._chapter_sort_key,
        )
        total = 0
        for fpath in chapter_files:
            m = re.search(r"第(\d+)章", os.path.basename(fpath))
            if not m:
                continue
            ch = int(m.group(1))
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            self.index_chapter(ch, text)
            total += 1
        return total

    def get_stats(self) -> dict:
        self._ensure_collection()
        if self._collection is None:
            return {"total_chunks": 0, "collection_name": "chapters", "novel": self.novel_name, "embedding_model": config.VECTOR_MEMORY_MODEL}
        count = self._collection.count()
        return {
            "total_chunks": count,
            "collection_name": "chapters",
            "novel": self.novel_name,
            "embedding_model": config.VECTOR_MEMORY_MODEL,
        }

    def has_index(self) -> bool:
        self._ensure_collection()
        if self._collection is None:
            return False
        return self._collection.count() > 0

    def _split_paragraphs(self, text: str, chapter_num: int) -> list[dict]:
        raw = text.split("\n\n")
        result = []
        seen = set()
        for i, para in enumerate(raw):
            para = para.strip()
            if len(para) < 50:
                continue
            para = para[:800]
            pid = f"ch{chapter_num:03d}_p{i:04d}"
            if pid in seen:
                continue
            seen.add(pid)
            result.append({"id": pid, "text": para, "index": i})
        return result

    @staticmethod
    def _chapter_sort_key(fpath: str) -> int:
        m = re.search(r"第(\d+)章", os.path.basename(fpath))
        return int(m.group(1)) if m else 9999
