import json
import os
import re
import time
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

import config
from novel_memory import NovelMemory
from novelist import _build_llm
from token_tracker import tracker


REVIEWER_SYSTEM = """你是一个小说细节一致性审查员。
你会收到：
1. 刚写好的章节正文
2. 从向量记忆中检索到的相关原文片段（可能涉及角色外貌、物品特征、场景描写）

你的任务是：
1. 对比新正文和原文片段，检查是否存在**细节矛盾**
2. 所谓矛盾是指：同一个物品被写成了不同的样子、同一个角色的外貌特征前后不一、
   场景中的固定特征（如学院食堂的老槐树、青藤廊道）被写错
3. 忽略风格差异和叙事节奏问题，只关注"写错了"的事实性矛盾

以JSON格式输出，只输出JSON不要任何其他文字：
{
  "has_conflict": true,
  "conflicts": [
    {
      "type": "外貌矛盾/物品矛盾/场景矛盾/年级矛盾/其他",
      "detail": "具体的矛盾描述",
      "new_text": "新正文中的写法（引用原文）",
      "original_text": "原文中的写法（引用原文）",
      "suggestion": "建议改成什么"
    }
  ],
  "summary": "一句话总结审查结果"
}
如果没有矛盾，conflicts 为空数组。"""


def _extract_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    if not usage:
        usage = {}
    input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
    output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
    cache_hit_tokens = (
        usage.get("cache_read_input_tokens", 0)
        or usage.get("prompt_cache_hit_tokens", 0)
        or usage.get("cached_tokens", 0)
    )
    if not cache_hit_tokens and input_tokens:
        cache_miss = usage.get("prompt_cache_miss_tokens", 0) or usage.get("cache_creation_input_tokens", 0)
        if cache_miss and cache_miss < input_tokens:
            cache_hit_tokens = input_tokens - cache_miss
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_hit_tokens": cache_hit_tokens,
    }


def check_consistency(chapter_text: str, chapter_num: int, max_retries: int = 1) -> dict:
    if not config.VECTOR_MEMORY_ENABLED:
        return {"has_conflict": False, "conflicts": [], "summary": "记忆未启用"}

    try:
        memory = NovelMemory(config.ACTIVE_NOVEL)
        if not memory.has_index():
            return {"has_conflict": False, "conflicts": [], "summary": "记忆库为空"}
    except Exception:
        return {"has_conflict": False, "conflicts": [], "summary": "无法访问记忆库"}

    queries = _extract_search_queries(chapter_text)
    if not queries:
        return {"has_conflict": False, "conflicts": [], "summary": "无可审查的关键词"}

    all_snippets = []
    for q in queries[:5]:
        try:
            results = memory.search(q, k=3)
            all_snippets.extend(results)
        except Exception:
            continue

    seen = set()
    snippets = []
    for s in all_snippets:
        if s["chapter"] == chapter_num:
            continue
        key = s["text"][:100]
        if key not in seen:
            seen.add(key)
            snippets.append(s)

    if not snippets:
        return {"has_conflict": False, "conflicts": [], "summary": "未检索到相关原文"}

    recall_text = "\n\n".join([
        f"[第{r['chapter']}章] {r['text'][:500]}"
        for r in snippets[:10]
    ])

    human = (
        f"【新章节正文（第{chapter_num}章）】\n"
        f"{chapter_text[:4000]}\n\n"
        f"【相关原文片段】\n"
        f"{recall_text}"
    )

    llm = _build_llm(config.EXTRACTOR_CONFIG)
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            result = llm.invoke([
                SystemMessage(content=REVIEWER_SYSTEM),
                HumanMessage(content=human),
            ])
            raw = result.content.strip()
            cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
            review = json.loads(cleaned)

            usage = _extract_usage(result)
            tracker.add_usage(
                agent_name="reviewer",
                model=config.EXTRACTOR_CONFIG["model"],
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_hit_tokens=usage.get("cache_hit_tokens", 0),
                is_pro=False,
            )

            return review
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(2)

    return {
        "has_conflict": False,
        "conflicts": [],
        "summary": f"审查失败: {last_error}",
    }


def _extract_search_queries(text: str) -> list[str]:
    keywords = set()

    text_clean = re.sub(r'[「」『』""（）()]', ' ', text)

    char_names = _get_character_names()

    for name in char_names:
        if name in text_clean or text_clean.find(name) != -1:
            keywords.add(f"{name}的描写")
            keywords.add(f"{name}的外貌")
            keywords.add(f"{name}的装备")

    for m in re.finditer(r'(?:银色|白色|黑色|红色|蓝色|绿色|灰色|破旧|崭新|旧)[的\s]?[\u4e00-\u9fff]{1,4}', text_clean):
        phrase = m.group().strip()
        if len(phrase) >= 2:
            keywords.add(phrase)

    for m in re.finditer(r'[\u4e00-\u9fff]{2,8}(?:的)?(?:描写|样子|外貌|装备|武器|魔能体|手套|带子)', text_clean):
        keywords.add(m.group().strip())

    return list(keywords)[:10]


def _get_character_names() -> list[str]:
    names = set()
    try:
        import config as cfg
        state_path = os.path.join(cfg.get_novel_knowledge_dir(), "state.json")
        if os.path.exists(state_path):
            import json as _json
            with open(state_path, "r", encoding="utf-8") as f:
                state = _json.load(f)
            names.update(state.get("characters", {}).keys())

        chars_path = os.path.join(cfg.get_novel_knowledge_dir(), "角色.md")
        if os.path.exists(chars_path):
            with open(chars_path, "r", encoding="utf-8") as f:
                content = f.read()
            for m in re.finditer(r'^### (.+?)$', content, re.MULTILINE):
                name = m.group(1).strip()
                if name:
                    names.add(name)
    except Exception:
        pass
    return list(names)
