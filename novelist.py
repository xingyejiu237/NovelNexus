import json
import os
import re
import time
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import config
from system_builder import (
    build_writer_prompt, build_writer_human,
    build_discussion_prompt, build_extractor_prompt,
    build_compressor_prompt, estimate_tokens,
    RECALL_PLANNER_PROMPT, CONFIRMATION_PROMPT,
)
from state_manager import (
    load_state, save_state, load_summaries, save_summaries,
    append_summary, save_snapshot, save_output_chapter,
    save_discussion_snapshot_pre, save_discussion_snapshot_post,
    save_discussion_log, load_discussion_log,
    read_knowledge_file, write_knowledge_file,
    record_event, get_summary_text,
)
from token_tracker import tracker


def _build_llm(config_dict: dict) -> ChatOpenAI:
    kwargs = {
        "model": config_dict["model"],
        "api_key": config.DEEPSEEK_API_KEY,
        "base_url": config.DEEPSEEK_BASE_URL,
        "temperature": config_dict.get("temperature", 1.0),
        "max_tokens": config_dict.get("max_tokens", 6000),
    }
    if config_dict.get("top_p") is not None and config_dict["top_p"] != 1.0:
        kwargs["top_p"] = config_dict["top_p"]
    if config_dict.get("frequency_penalty"):
        kwargs["frequency_penalty"] = config_dict["frequency_penalty"]
    if config_dict.get("presence_penalty"):
        kwargs["presence_penalty"] = config_dict["presence_penalty"]
    model_kwargs = {}
    if config_dict.get("response_format"):
        model_kwargs["response_format"] = config_dict["response_format"]
    if config_dict.get("enable_thinking"):
        kwargs.pop("temperature", None)
        model_kwargs["extra_body"] = {
            "thinking": {
                "type": "enabled",
                "reasoning_effort": config_dict.get("reasoning_effort", "medium"),
            }
        }
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return ChatOpenAI(**kwargs)


_novelist_llm: Optional[ChatOpenAI] = None
_extractor_llm: Optional[ChatOpenAI] = None
_compressor_llm: Optional[ChatOpenAI] = None


def _get_novelist_llm() -> ChatOpenAI:
    global _novelist_llm
    if _novelist_llm is None:
        _novelist_llm = _build_llm(config.NOVELIST_CONFIG)
    return _novelist_llm


def _get_extractor_llm() -> ChatOpenAI:
    global _extractor_llm
    if _extractor_llm is None:
        _extractor_llm = _build_llm(config.EXTRACTOR_CONFIG)
    return _extractor_llm


def _get_compressor_llm() -> ChatOpenAI:
    global _compressor_llm
    if _compressor_llm is None:
        _compressor_llm = _build_llm(config.COMPRESSOR_CONFIG)
    return _compressor_llm


def _count_chinese_chars(text: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef0-9]', text))


def _extract_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    extra_usage = getattr(response, "response_metadata", {}).get("token_usage", {}) or {}
    if extra_usage and usage:
        merged = dict(usage)
        for k, v in extra_usage.items():
            if k not in merged and v is not None:
                merged[k] = v
        usage = merged
    elif extra_usage and not usage:
        usage = extra_usage

    input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
    output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

    cache_hit_tokens = (
        usage.get("cache_read_input_tokens", 0)
        or usage.get("prompt_cache_hit_tokens", 0)
        or usage.get("cached_tokens", 0)
    )
    if not cache_hit_tokens and input_tokens:
        cache_miss = (
            usage.get("prompt_cache_miss_tokens", 0)
            or usage.get("cache_creation_input_tokens", 0)
        )
        if cache_miss and cache_miss < input_tokens:
            cache_hit_tokens = input_tokens - cache_miss

    raw_keys = list(usage.keys())[:12] if usage else []
    if raw_keys:
        print(f"  [usage] input={input_tokens} output={output_tokens} cache_hit={cache_hit_tokens} keys={raw_keys}")

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "_raw_usage_keys": raw_keys,
    }


TITLE_CHECK_SYSTEM = """你是一个小说标题审查员。判断标题是否准确概括了本章核心内容。
规则：
- 标题不需要面面俱到，但不能完全不相关
- 标题可以诗意、可以隐喻，但要和本章内容有明确关联
- 如果标题是"第X章 训练/测试/日常"这种通用词，也算匹配（虽然没有亮点，但没有错误）

以JSON格式输出，只输出JSON：
{"match": true}
或
{"match": false, "suggested_title": "建议的新标题"}
如果不匹配，给出的新标题要简洁、有内容关联、有信息量。不要用通用标题。"""


def _check_title(chapter_text: str) -> dict:
    m = re.search(r'^#\s*第[\d零一二三四五六七八九十百千]+章\s+(.+)$', chapter_text, re.MULTILINE)
    if not m:
        return None
    title = m.group(1).strip()
    if not title:
        return None

    body = chapter_text[:3000]
    prompt = f"标题：{title}\n\n正文开头：\n{body}\n\n判断标题是否匹配正文。"
    llm = _get_extractor_llm()
    try:
        result = llm.invoke([SystemMessage(content=TITLE_CHECK_SYSTEM), HumanMessage(content=prompt)])
        raw = result.content.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        review = json.loads(cleaned)
    except Exception as e:
        print(f"⚠️ [_check_title] JSON解析失败: {e}")
        return None

    usage_data = _extract_usage(result)
    tracker.add_usage(
        agent_name="checker",
        model=config.EXTRACTOR_CONFIG["model"],
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        cache_hit_tokens=usage_data.get("cache_hit_tokens", 0),
        is_pro=False,
    )

    if not review.get("match") and review.get("suggested_title"):
        new_title = review["suggested_title"].strip().strip("「」""''")
        if new_title and len(new_title) <= 20:
            chapter_text = re.sub(
                r'^(#\s*第[\d零一二三四五六七八九十百千]+章\s+).+$',
                r'\1' + new_title,
                chapter_text,
                count=1,
                flags=re.MULTILINE,
            )
            return {"original_title": title, "new_title": new_title, "rewritten": True}

    return {"original_title": title, "match": review.get("match", True), "rewritten": False}


def _confirm_fact(query: str) -> str:
    result = ""

    # ── Stage 1: 向量语义搜索 + LLM 二次确认 ──
    snippets = []
    try:
        from novel_memory import NovelMemory
        memory = NovelMemory(config.ACTIVE_NOVEL, lazy=True)
        if memory.has_index():
            snippets = memory.search(query, k=3)
    except Exception:
        pass

    if snippets:
        snippet_text = "\n".join(
            f"{i+1}. [第{r['chapter']}章] {r['text'][:300]}"
            for i, r in enumerate(snippets)
        )
        try:
            llm = _get_extractor_llm()
            resp = llm.invoke([
                SystemMessage(content=CONFIRMATION_PROMPT.format(query=query, snippets=snippet_text)),
                HumanMessage(content="请判断以上片段是否回答了查询。"),
            ])
            raw = resp.content.strip()
            cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            check = json.loads(cleaned)
            if check.get("found") and check.get("conclusion"):
                ch = check.get("chapter", "")
                result = f"{check['conclusion']} [第{ch}章]" if ch else check["conclusion"]
        except Exception as e:
            print(f"⚠️ [_confirm_fact stage1] 确认失败: {e}")

    if result:
        return result

    # ── Stage 2: 搜索 summaries.json ──
    try:
        summaries = load_summaries()
        query_keywords = set(re.sub(r'[，。？、；：""''（）()]', ' ', query).split())
        best_match = None
        best_score = 0
        for s in summaries:
            summary_text = s.get("summary", "")
            summary_keywords = set(re.sub(r'[，。？、；：""''（）()]', ' ', summary_text).split())
            overlap = query_keywords & summary_keywords
            score = len(overlap)
            if score > best_score:
                best_score = score
                best_match = s
        if best_match and best_score >= 1:
            result = f"{best_match['summary'][:200]} [第{best_match['chapter']}章摘要]"
    except Exception as e:
        print(f"⚠️ [_confirm_fact stage2] 摘要检索失败: {e}")

    if result:
        return result

    # ── Stage 3: 读对应章节原文全文 ──
    try:
        from novel_memory import NovelMemory
        memory = NovelMemory(config.ACTIVE_NOVEL, lazy=True)
        if memory.has_index():
            results = memory.search(query, k=1)
            if results:
                target_ch = results[0]["chapter"]
                ch_path = os.path.join(config.get_novel_output_dir(), f"第{target_ch:03d}章.md")
                if os.path.exists(ch_path):
                    with open(ch_path, "r", encoding="utf-8") as f:
                        full_text = f.read()
                    for line in full_text.split("\n"):
                        if len(line) > 20 and any(kw in line for kw in query.split()):
                            result = f"{line.strip()[:200]} [第{target_ch}章原文]"
                            break
                    if not result:
                        result = f"在第{target_ch}章中未找到明确结论"
    except Exception as e:
        print(f"⚠️ [_confirm_fact stage3] 原文查找失败: {e}")

    return result


def _plan_and_recall(state: dict, user_direction: str, chapter_num: int) -> dict:
    result = {
        "plan": "",
        "confirmed_facts": "",
        "snippets": [],
    }

    if chapter_num <= 1:
        return result
    if not config.VECTOR_MEMORY_ENABLED:
        return result

    context_parts = []
    if state.get("current_scene"):
        context_parts.append(f"当前场景：{state['current_scene']}")
    if state.get("current_time"):
        context_parts.append(f"当前时间：{state['current_time']}")
    if state.get("current_suspense"):
        context_parts.append(f"当前悬念：{state['current_suspense']}")
    char_names = list(state.get("characters", {}).keys())
    if char_names:
        context_parts.append(f"角色：{'、'.join(char_names[:8])}")
    if user_direction:
        context_parts.append(f"作者方向：{user_direction}")
    context = "\n".join(context_parts)

    llm = _get_extractor_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=RECALL_PLANNER_PROMPT),
            HumanMessage(content=f"第{chapter_num}章写作规划。当前状态：\n\n{context}"),
        ])
        raw = response.content.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        plan = json.loads(cleaned)
    except Exception as e:
        print(f"⚠️ [_plan_and_recall] 规划失败: {e}")
        return result

    tracker.add_usage(
        agent_name="recall_planner",
        model=config.EXTRACTOR_CONFIG["model"],
        input_tokens=0,
        output_tokens=0,
        cache_hit_tokens=0,
        is_pro=False,
    )

    result["plan"] = plan.get("plan", "")

    need_confirm = plan.get("need_confirm", [])
    fact_lines = []
    for item in need_confirm:
        typ = item.get("type", "")
        if typ == "equipment":
            for name in item.get("characters", []):
                if name in state.get("characters", {}):
                    eq = state["characters"][name].get("equipment", {})
                    if eq:
                        eq_str = ", ".join(f"{k}={v}" for k, v in eq.items())
                        fact_lines.append(f"  · {name}的装备：{eq_str}")
        elif typ == "fact":
            query = item.get("query", "")
            if query:
                confirmation = _confirm_fact(query)
                if confirmation:
                    fact_lines.append(f"  · {query} → {confirmation}")
                else:
                    fact_lines.append(f"  · {query} → 未能在已有章节中找到明确答案")
        elif typ == "scene":
            query = item.get("query", "")
            if query:
                try:
                    from novel_memory import NovelMemory
                    memory = NovelMemory(config.ACTIVE_NOVEL, lazy=True)
                    if memory.has_index():
                        results = memory.search(query, k=1)
                        if results:
                            r = results[0]
                            fact_lines.append(f"  · {query} → [第{r['chapter']}章] {r['text'][:200]}")
                except Exception:
                    pass

    if fact_lines:
        result["confirmed_facts"] = "\n".join(fact_lines)

    vector_queries = plan.get("vector_queries", [])
    if vector_queries:
        try:
            from novel_memory import NovelMemory
            memory = NovelMemory(config.ACTIVE_NOVEL, lazy=True)
            if memory.has_index():
                all_snippets = []
                for query in vector_queries[:5]:
                    results = memory.search(query, k=3)
                    all_snippets.extend(results)
                all_snippets.sort(key=lambda x: x["chapter"])
                seen = set()
                snippets = []
                for s in all_snippets:
                    key = (s["chapter"], s["paragraph_index"])
                    if key not in seen:
                        seen.add(key)
                        snippets.append(s)
                result["snippets"] = snippets[:config.VECTOR_MEMORY_SEARCH_K * 3]
        except Exception as e:
            print(f"⚠️ [_plan_and_recall] 向量检索失败: {e}")

    return result


def generate_chapter(user_direction: str = "",
                     is_rewrite: bool = False,
                     rewrite_reason: str = "",
                     recall_queries: list[str] = None,
                     rewrite_history: list[str] = None) -> dict:
    state = load_state()
    if is_rewrite:
        chapter_num = max(1, state["next_chapter"] - 1)
    else:
        chapter_num = state["next_chapter"]
    alerts: list[str] = []

    system = build_writer_prompt(state)
    if estimate_tokens(system) > config.CONTEXT_SOFT_LIMIT_TOKENS:
        summary_text = get_summary_text(current_volume_only=True)
        system = "\n\n".join(system.split("\n\n")[:6]) + "\n\n" + summary_text

    summaries = load_summaries()
    recent_raw = []
    for s in summaries[-config.RECENT_CHAPTERS_COUNT:]:
        ch = s["chapter"]
        out_path = os.path.join(config.get_novel_output_dir(), f"第{ch:03d}章.md")
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                recent_raw.append(f.read())
    recent_chapters_text = "\n\n---\n\n".join(recent_raw)

    # ── 回忆规划层：先规划再回忆（V4新增） ──
    recall_plan = None
    writing_plan_text = ""
    confirmed_facts_text = ""
    if config.VECTOR_MEMORY_ENABLED and chapter_num > 1:
        recall_plan = _plan_and_recall(state, user_direction, chapter_num)
        if recall_plan and recall_plan.get("plan"):
            writing_plan_text = recall_plan["plan"]
            confirmed_facts_text = recall_plan.get("confirmed_facts", "")

    # ── 规划层提供的向量片段 ──
    recall_snippets = None
    if recall_plan and recall_plan.get("snippets"):
        recall_snippets = recall_plan["snippets"]

    # ── 规划层回退：如果规划失败或为空，用旧式自动检索 ──
    if not recall_snippets and config.VECTOR_MEMORY_ENABLED and chapter_num > 1:
        try:
            from novel_memory import NovelMemory
            memory = NovelMemory(config.ACTIVE_NOVEL)
            if memory.has_index():
                auto_queries = []
                char_names = list(state.get("characters", {}).keys())
                if char_names:
                    for name in char_names[:10]:
                        auto_queries.append(f"关于{name}的描写、外貌、行为、对话、动作")
                        auto_queries.append(f"{name}的手套、魔能体、装备、感应线、等级")
                if state.get("current_scene"):
                    auto_queries.append(state["current_scene"])
                if auto_queries:
                    all_snippets = []
                    for query in auto_queries[:6]:
                        results = memory.search(query, k=3)
                        all_snippets.extend(results)
                    all_snippets.sort(key=lambda x: x["chapter"])
                    seen = set()
                    fallback_snippets = []
                    for s in all_snippets:
                        key = (s["chapter"], s["paragraph_index"])
                        if key not in seen:
                            seen.add(key)
                            fallback_snippets.append(s)
                    recall_snippets = fallback_snippets[:config.VECTOR_MEMORY_SEARCH_K * 3]
        except Exception as e:
            alerts.append(f"自动检索失败: {e}")

    # ── 用户手动检索覆盖 ──
    if recall_queries and config.VECTOR_MEMORY_ENABLED and chapter_num > 1:
        try:
            from novel_memory import NovelMemory
            memory = NovelMemory(config.ACTIVE_NOVEL)
            if memory.has_index():
                manual_snippets = []
                for query in recall_queries:
                    results = memory.search(query, k=config.VECTOR_MEMORY_SEARCH_K)
                    manual_snippets.extend(results)
                manual_snippets.sort(key=lambda x: x["chapter"])
                seen = set()
                recall_snippets = []
                for s in manual_snippets:
                    key = (s["chapter"], s["paragraph_index"])
                    if key not in seen:
                        seen.add(key)
                        recall_snippets.append(s)
                recall_snippets = recall_snippets[:config.VECTOR_MEMORY_SEARCH_K * 2]
        except Exception as e:
            alerts.append(f"手动检索失败: {e}")

    human = build_writer_human(
        state=state,
        user_direction=user_direction,
        rewrite_reason=rewrite_reason,
        recent_chapters_text=recent_chapters_text,
        recall_snippets=recall_snippets,
        writing_plan=writing_plan_text,
        confirmed_facts=confirmed_facts_text,
        rewrite_history=rewrite_history or [],
        chapter_num=chapter_num,
    )

    llm = _get_novelist_llm()
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    except Exception as e:
        print(f"❌ [generate] LLM调用失败: {e}")
        time.sleep(3)
        try:
            result = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        except Exception as e2:
            return {"error": f"两次重试均失败: {e2}"}

    text = result.content.strip()
    usage_data = _extract_usage(result)
    tracker.add_usage(
        agent_name="novelist",
        model=config.NOVELIST_CONFIG["model"],
        input_tokens=usage_data["input_tokens"],
        output_tokens=usage_data["output_tokens"],
        cache_hit_tokens=usage_data["cache_hit_tokens"],
        is_pro="pro" in config.NOVELIST_CONFIG.get("model", ""),
    )

    # 分离 Agent 附言（正文结尾 `---` 分隔线之后的内容）
    agent_suggestion = ""
    writer_questions = ""
    sep_match = re.search(r'\n---\s*\n\s*💬 Agent附言[：:]\s*\n', text)
    if sep_match:
        sep_end = sep_match.end()
        agent_suggestion = text[sep_end:].strip()
        text = text[:sep_match.start()]
        # 从附言中分离可选的下章方向疑问
        qs_match = re.search(r'📋 下章方向疑问[：:]\s*\n', agent_suggestion)
        if qs_match:
            writer_questions = agent_suggestion[qs_match.end():].strip()
            agent_suggestion = agent_suggestion[:qs_match.start()].strip()
    # 如果正文末尾也有一条独立的 `---`（没有附言标识），不管它
    volume_switch = None
    conflict_alert = None

    if config.VECTOR_MEMORY_ENABLED and chapter_num > 2 and len(text) > 200:
        try:
            from consistency_checker import check_consistency
            review = check_consistency(text, chapter_num)
            if review.get("has_conflict") and review.get("conflicts"):
                conflict_lines = []
                for c in review["conflicts"][:3]:
                    conflict_lines.append(
                        f"⚠️ {c.get('type', '潜在矛盾')}：{c.get('detail', '')}\n"
                        f"  ├ 新章写法：{c.get('new_text', '')}\n"
                        f"  ├ 原文写法：{c.get('original_text', '')}\n"
                        f"  └ 💡 建议：{c.get('suggestion', '')}"
                    )
                conflict_alert = "\n\n".join(conflict_lines)
        except Exception as e:
            alerts.append(f"一致性审查失败: {e}")

    # ── 标题合理性审查（V4新增） ──
    title_review = None
    if chapter_num > 1 and len(text) > 200:
        try:
            title_review = _check_title(text)
        except Exception as e:
            alerts.append(f"标题审查失败: {e}")

    return {
        "chapter_text": text,
        "agent_suggestion": agent_suggestion,
        "writer_questions": writer_questions,
        "alerts": alerts,
        "usage_info": usage_data,
        "chapter_num": chapter_num,
        "volume_switch": volume_switch,
        "conflict_alert": conflict_alert,
        "title_review": title_review,
    }


def accept_chapter(chapter_text: str,
                   agent_suggestion: str,
                   usage_info: dict,
                   volume_switch: Optional[dict],
                   chapter_num: int = None) -> dict:
    state = load_state()
    if chapter_num is None:
        chapter_num = state["next_chapter"]

    extract_system, extract_human = build_extractor_prompt(chapter_text, state)
    llm = _get_extractor_llm()
    extract_result = None
    try:
        extract_result = llm.invoke([SystemMessage(content=extract_system), HumanMessage(content=extract_human)])
        raw = extract_result.content.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        extracted = json.loads(cleaned)
    except Exception as e:
        print(f"⚠️ [extract] JSON解析失败: {e}")
        extracted = {}
    usage_data = _extract_usage(extract_result) if extract_result is not None else {}
    tracker.add_usage(
        agent_name="extractor",
        model=config.EXTRACTOR_CONFIG["model"],
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        cache_hit_tokens=usage_data.get("cache_hit_tokens", 0),
        is_pro="pro" in config.EXTRACTOR_CONFIG.get("model", ""),
    )

    summary = extracted.get("summary", "") or "（摘要提取失败）"
    append_summary(chapter_num, summary)

    if volume_switch:
        state["volume_history"].append({
            "num": state["current_volume"]["num"],
            "title": state["current_volume"]["title"],
            "summary": f"第{state['current_volume']['num']}卷完",
            "chapters": f"{state.get('_vol_start_chapter', 1)}-{chapter_num}",
        })
        state["current_volume"] = volume_switch
        state["_vol_start_chapter"] = chapter_num

    location = extracted.get("location", "")
    if location:
        state["current_scene"] = location
    if extracted.get("time_passage"):
        new_time = extracted["time_passage"]
        if re.match(r'^第\d+卷第\d+天$', new_time):
            state["current_time"] = new_time
    if extracted.get("suspense"):
        state["current_suspense"] = extracted["suspense"]
        state["suspense"].append({"content": extracted["suspense"], "chapter": chapter_num, "resolved": False})
    if extracted.get("characters"):
        _is_placeholder = re.compile(r'^.{2,}(?:学员|的男人|的女人|的人|女生|路人)$')
        _new_named = set()
        for name, info in extracted["characters"].items():
            if name not in state["characters"]:
                state["characters"][name] = {}
            state["characters"][name].update(info)
            if not _is_placeholder.match(name):
                _new_named.add(name)
        if _new_named:
            to_remove = []
            for old_name in list(state["characters"].keys()):
                if _is_placeholder.match(old_name):
                    old_loc = state["characters"][old_name].get("location", "")
                    for new_name in _new_named:
                        new_loc = state["characters"][new_name].get("location", "")
                        if old_loc and new_loc and old_loc == new_loc:
                            to_remove.append(old_name)
                            break
            for name in to_remove:
                del state["characters"][name]
    if extracted.get("foreshadowing_new"):
        for f in extracted["foreshadowing_new"]:
            state["foreshadowing"]["pending"].append({
                "item": f["item"],
                "status": "未揭示",
                "chapter": chapter_num,
                "keywords": f.get("keywords", []),
                "importance": 3,
            })
    if extracted.get("foreshadowing_resolved"):
        for resolved_item in extracted["foreshadowing_resolved"]:
            resolved_keywords = set(resolved_item.get("keywords", []))
            resolved_name = resolved_item.get("item", "")
            resolved_name_clean = resolved_name.replace(" ", "").replace("的", "")
            match = None
            for pf in state["foreshadowing"]["pending"]:
                pf_keywords = set(pf.get("keywords", []))
                if resolved_keywords and pf_keywords and resolved_keywords & pf_keywords:
                    match = pf
                    break
                pf_name_clean = pf.get("item", "").replace(" ", "").replace("的", "")
                if resolved_name_clean and pf_name_clean:
                    if resolved_name_clean in pf_name_clean or pf_name_clean in resolved_name_clean:
                        match = pf
                        break
            if match:
                state["foreshadowing"]["resolved"].append({"item": match["item"], "chapter": chapter_num})
                state["foreshadowing"]["pending"].remove(match)
    if extracted.get("new_structures"):
        _append_structures_to_dictionary(extracted["new_structures"], chapter_num)
    if extracted.get("new_characters"):
        _append_characters_to_roster(extracted["new_characters"])
    if extracted.get("physical_details"):
        try:
            from state_manager import append_physical_details
            apd = append_physical_details(extracted["physical_details"])
            if apd > 0:
                print(f"📝 细节档案：新增{apd}条外貌/物品记录")
        except Exception as e:
            print(f"⚠️ [physical details] 保存失败: {e}")
    if extracted.get("equipment_updates"):
        for name, eq_dict in extracted["equipment_updates"].items():
            if name in state["characters"]:
                existing_eq = state["characters"][name].get("equipment", {})
                existing_eq.update(eq_dict)
                state["characters"][name]["equipment"] = existing_eq
    if extracted.get("tension"):
        recent = state.get("_recent_tension", [])
        recent.append(extracted["tension"])
        if len(recent) > 10:
            recent = recent[-10:]
        state["_recent_tension"] = recent
    if extracted.get("scene_catalog_entry") and extracted["scene_catalog_entry"].get("type"):
        entry = extracted["scene_catalog_entry"]
        existing = [s for s in state.get("scene_catalog", []) if s["type"] == entry["type"]]
        if existing:
            existing[0]["appearances"].append(chapter_num)
            if entry.get("effectiveness"):
                existing[0]["effectiveness"] = entry["effectiveness"]
        else:
            entry["appearances"] = [chapter_num]
            state.setdefault("scene_catalog", []).append(entry)
            state["scene_catalog"] = state["scene_catalog"][-30:]
    if extracted.get("used_imagery"):
        state["_last_imagery"] = extracted["used_imagery"]
    if extracted.get("timeline_events"):
        for te in extracted["timeline_events"]:
            if isinstance(te, dict):
                state["timeline_events"].append(te)
    suspense_types = state.get("章节悬念类型历史", [])
    if extracted.get("suspense"):
        stype = extracted.get("scene_type", "通用")
        suspense_types.append(stype)
    state["章节悬念类型历史"] = suspense_types[-10:]

    state["next_chapter"] = chapter_num + 1
    save_snapshot(chapter_num, state, load_summaries())
    save_state(state)
    save_output_chapter(chapter_num, chapter_text)
    record_event("chapter", {"chapter": chapter_num})

    if config.VECTOR_MEMORY_ENABLED:
        try:
            from novel_memory import NovelMemory
            memory = NovelMemory(config.ACTIVE_NOVEL)
            memory.index_chapter(chapter_num, chapter_text)
        except Exception as e:
            print(f"⚠️ [vector index] 章节索引失败: {e}")

    return {
        "success": True,
        "message": f"第{chapter_num}章已保存",
        "next_chapter": state["next_chapter"],
    }


def _append_structures_to_dictionary(structures: list[dict], chapter_num: int):
    try:
        rules_path = os.path.join(config.get_novel_knowledge_dir(), "续写规范.md")
        if not os.path.exists(rules_path):
            return
        with open(rules_path, "r", encoding="utf-8") as f:
            content = f.read()
        dict_section = re.search(r'## 已命名结构词典\n+.*?\n+((?:\|.*\n)+)', content, re.DOTALL)
        if not dict_section:
            return
        existing_table = dict_section.group(1)
        existing_names = set(re.findall(r'^\| +([^ |]+) +\|', existing_table, re.MULTILINE))
        new_lines = []
        for s in structures:
            name = s.get("name", "").strip()
            desc = s.get("desc", "").strip()
            if not name or not desc:
                continue
            if name in existing_names:
                continue
            new_lines.append(f"| {name} | 第{chapter_num}章 | {desc} |")
            existing_names.add(name)
        if not new_lines:
            return
        insert_pos = dict_section.end()
        new_content = content[:insert_pos] + "\n".join(new_lines) + "\n" + content[insert_pos:]
        with open(rules_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        print(f"⚠️ [结构词典更新失败] {e}")


def _append_characters_to_roster(characters: list[dict]):
    try:
        chars_path = os.path.join(config.get_novel_knowledge_dir(), "角色.md")
        if not os.path.exists(chars_path):
            return
        with open(chars_path, "r", encoding="utf-8") as f:
            content = f.read()
        existing_names = set(re.findall(r'^### (.+?)$', content, re.MULTILINE))
        new_blocks = []
        for c in characters:
            name = c.get("name", "").strip()
            role = c.get("role", "").strip()
            desc = c.get("desc", "").strip()
            if not name or not role or not desc:
                continue
            if name in existing_names:
                continue
            block = f"### {name}\n\n- {role}\n- **描述**：{desc}\n"
            new_blocks.append(block)
            existing_names.add(name)
        if not new_blocks:
            return
        insert_marker = "如果你发现遗漏或描述不准确，可以通过讨论模式手动修正。\n\n"
        insert_pos = content.find(insert_marker)
        if insert_pos == -1:
            return
        insert_pos += len(insert_marker)
        new_content = content[:insert_pos] + "\n".join(new_blocks) + "\n" + content[insert_pos:]
        with open(chars_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        print(f"⚠️ [角色追加失败] {e}")


def discuss(user_message: str, chat_history: list = None) -> dict:
    state = load_state()
    recall_snippets = None
    if config.VECTOR_MEMORY_ENABLED:
        recall_triggers = ["样子", "什么颜色", "长什么样", "之前", "第几章", "回忆", "描写", "外貌", "装备", "武器", "细节", "特征", "什么样", "还记得"]
        should_recall = any(t in user_message for t in recall_triggers)
        if should_recall:
            try:
                from novel_memory import NovelMemory
                memory = NovelMemory(config.ACTIVE_NOVEL)
                if memory.has_index():
                    recall_snippets = memory.search(user_message, k=config.VECTOR_MEMORY_SEARCH_K * 2)
            except Exception as e:
                print(f"⚠️ [discuss recall] 检索失败: {e}")

    system = build_discussion_prompt(state, recall_snippets=recall_snippets)

    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        lines = ["【对话历史】"]
        for msg in recent:
            role = "作者" if msg["role"] == "user" else "你"
            content = msg.get("content", "")
            if content:
                lines.append(f"{role}：{content[:300]}")
        history_text = "\n".join(lines) + "\n\n"

    human = history_text + f"作者的想法：{user_message}\n\n请分析影响并提供方案。"

    llm = _get_novelist_llm()
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    except Exception as e:
        return {"error": f"讨论失败: {e}"}

    text = result.content.strip()
    usage_data = _extract_usage(result)
    tracker.add_usage(
        agent_name="discussion",
        model=config.NOVELIST_CONFIG["model"],
        input_tokens=usage_data["input_tokens"],
        output_tokens=usage_data["output_tokens"],
        cache_hit_tokens=usage_data["cache_hit_tokens"],
        is_pro="pro" in config.NOVELIST_CONFIG.get("model", ""),
    )

    modification_plan = None
    try:
        cleaned = text
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            json_str = cleaned[first_brace:last_brace + 1]
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            json_str = json_str.replace('，', ',').replace('：', ':')
            data = json.loads(json_str)
            if "modification_plan" in data:
                modification_plan = data["modification_plan"]
                text = data.get("reply", text)
            elif "analyses" in data or "options" in data:
                modification_plan = data
    except (json.JSONDecodeError, Exception):
        pass

    return {
        "reply": text,
        "modification_plan": modification_plan,
    }


def execute_modifications(modifications: list[dict]) -> dict:
    disc_id = f"disc_{int(time.time())}"
    affected_files = list(set(m["file"] for m in modifications))
    save_discussion_snapshot_pre(disc_id, affected_files)

    for mod in modifications:
        file = mod["file"]
        action = mod.get("action", "append_section")
        content = read_knowledge_file(file)
        if action == "append_section":
            section_title = mod.get("section_title", "")
            new_content = mod.get("content", "")
            content += f"\n\n# {section_title}\n{new_content}"
        elif action == "replace_section":
            section_title = mod.get("section_title", "")
            new_content = mod.get("content", "")
            pattern = re.compile(rf'^#+\s*{re.escape(section_title)}\s*$', re.MULTILINE)
            m = pattern.search(content)
            if m:
                start = m.start()
                after_title = content[m.end():]
                next_heading = re.search(r'^#+ ', after_title, re.MULTILINE)
                if next_heading:
                    end_pos = m.end() + next_heading.start()
                else:
                    end_pos = len(content)
                content = content[:start] + new_content + content[end_pos:]
            else:
                content += f"\n\n{new_content}"
        elif action == "replace_content":
            old_text = mod.get("old_text", "")
            new_text = mod.get("new_text", "")
            if old_text and old_text in content:
                content = content.replace(old_text, new_text, 1)
        elif action == "replace_file":
            content = mod.get("content", "")
        elif action == "update_character" and "角色.md" in file:
            content += f"\n\n### {mod.get('character', '')}\n{mod.get('content', '')}"
        write_knowledge_file(file, content)

    save_discussion_snapshot_post(disc_id)

    log = load_discussion_log()
    log["sessions"].append({
        "id": disc_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trigger": modifications[0].get("section_title", "") if modifications else "",
        "summary": modifications[0].get("content", "")[:100] if modifications else "",
        "decisions": modifications,
        "applied": True,
    })
    save_discussion_log(log)
    record_event("discussion", {"disc_id": disc_id, "summary": modifications[0].get("content", "")[:100] if modifications else ""})

    return {"success": True, "message": f"已执行 {len(modifications)} 个修改", "disc_id": disc_id}


def init_novel(name: str, idea: str, protagonist: str) -> dict:
    from config import set_active_novel
    novel_dir = config.get_novel_dir(name)
    knowledge_dir = config.get_novel_knowledge_dir(name)
    output_dir = config.get_novel_output_dir(name)

    os.makedirs(knowledge_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(knowledge_dir, "snapshots"), exist_ok=True)

    for fname in ["世界观.md", "角色.md", "大纲.md", "风格.md", "续写规范.md"]:
        fpath = os.path.join(knowledge_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"# {fname.replace('.md', '')}\n\n*待填充*")

    initial_state = {
        "next_chapter": 1,
        "novel_title": name,
        "current_volume": {"num": 1, "title": "第一卷"},
        "current_scene": "",
        "current_time": "开始",
        "characters": {},
        "foreshadowing": {"pending": [], "resolved": []},
        "suspense": [],
        "current_suspense": "",
        "timeline_events": [],
        "volume_history": [],
        "self_calibrations": [],
        "health_report": {},
        "scene_catalog": [],
        "章节悬念类型历史": [],
        "_vol_start_chapter": 1,
    }
    with open(os.path.join(knowledge_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(initial_state, f, ensure_ascii=False, indent=2)
    for subfile in ["summaries.json", "discussion_log.json", "style_progression.json"]:
        subpath = os.path.join(knowledge_dir, subfile)
        if not os.path.exists(subpath):
            with open(subpath, "w", encoding="utf-8") as f:
                f.write("[]" if subfile in ("summaries.json",) else '{"sessions":[]}' if "discussion" in subfile else '{"volumes":[]}')

    events_path = os.path.join(knowledge_dir, "snapshots", "events.jsonl")
    if not os.path.exists(events_path):
        with open(events_path, "w", encoding="utf-8") as f:
            f.write("")

    set_active_novel(name)

    return {
        "success": True,
        "message": f"小说《{name}》已创建",
        "state": initial_state,
    }


_HOOKS = {
    "before_generate": [],
    "after_accept": [],
    "before_discuss": [],
    "after_modify": [],
}


def register_hook(event: str, callback: callable):
    if event in _HOOKS:
        _HOOKS[event].append(callback)
