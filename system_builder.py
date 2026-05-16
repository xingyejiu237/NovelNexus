import config
import re
from token_counter import count_tokens
from state_manager import (
    read_knowledge_file, load_state, get_summary_text,
    get_discussion_summary, load_style_progression,
)


WRITER_PERSONA = """你是小说的专职续写作者。
你每次只写一章，但你掌握本书全部设定和前情。
你的任务是写出高质量、有感染力且风格统一的章节正文。"""

WRITER_WORD_CONTROL = """【字数控制】
字数约 {word_min}-{word_max} 字，不得低于 {word_min} 字，不得超过 {word_max} 字。

【写作规范 — 写完必须自查】
1. ⛔ 搜"不是"二字，如有「不是X而是Y」句式全部改写为直接陈述
2. ⛔ 连续3个以上换行必须合并
3. ⛔ 检查比喻：全章不超过5处
4. ⛔ 章末悬念类型不得与最近3章重复
5. ⛔ 不得出现元叙事（打破第四面墙）
6. ⛔ 检查情绪：本章中每个有对话的角色至少有一次明确的情绪反应（表情/动作/语言/心理）。禁止全程"平静地说"。林野偶尔呆是正常的，但不能呆到对什么都没反应。

【输出格式 ⛔ 必须严格遵守】
正文写完后，另起一行输出 `---` 分隔线，换行后写「💬 Agent附言：」，然后写附言内容。
**注意：附言不是可选项，是必选项。章节结尾必须有 `---` 分隔线和 💬 Agent附言。**

【可选：下章方向疑问】
如果你对下一章的推进方向有不确定之处，可以在附言末尾增加「📋 下章方向疑问：」区块，提出1-3个具体问题。
作者会在下一章的「我的方向」中回答。非必选，不需要每章都提。
格式示例：
💬 Agent附言：
本章节奏适中，下章可适当加快。当前卷弧线未完成。
[矛盾提醒] xxx

📋 下章方向疑问：
Q1: 林野的手套是否应在下一章彻底报废？
Q2: 苏棠的T1魔能体是否应展示极限输出的副作用？

示例：
（正文最后一段结束。）

---
💬 Agent附言：
本章节奏适中，下章可适当加快。当前卷弧线未完成。
[矛盾提醒] xxx

⚠️ Agent附言必须在分隔线之后，不得写在正文段落里。"""

DISCUSSION_PERSONA = """你是这部小说的创意合伙人。
你完整掌握全部世界观、角色设定、故事大纲和当前进度。
当作者提出修改建议时，你的职责是：
- 分析该修改对现有世界观一致性的影响
- 分析对角色弧线和人物关系的影响
- 分析对已有大纲和伏笔布局的影响
- 提供 2-3 个具体方案供作者选择
- 指出潜在风险和落入俗套的可能性
- 像一个真正的合作者一样给出诚恳的创意判断，不一味附和"""

EXTRACTOR_SYSTEM = """你是一个小说章节分析器。分析章节正文，以JSON格式输出。
只输出JSON，不要任何其他文字。

JSON结构：
{
  "summary": "150字以内章节摘要",
  "location": "林野本章结束时的位置",
  "time_passage": "本章的时间递进。prompt中已给出上一章时间（如第1卷第3天），请据此判断：若本章与上一章在同一天，输出相同时间；若跨天（过夜/次日早上），天数+1。格式固定为第X卷第Y天，不要输出prompt中示例的值——示例只是说明规则",
  "characters": {"角色名": {"location": "位置", "state": "当前状态"}},
  "foreshadowing_new": [{"item": "新伏笔", "keywords": []}],
  "foreshadowing_resolved": [],
  "suspense": "当前悬念",
  "tension": 7,
  "scene_type": "对话/战斗/内心/探索/日常",
  "dialogue_ratio": 0.3,
  "timeline_events": [{"event": "事件", "when": "时间描述", "chapter": 0}],
  "used_imagery": ["核心意象"],
  "scene_catalog_entry": {"type": "场景类型", "characters": [], "effectiveness": "高/中/低", "signature_pattern": ""},
  "new_structures": [{"name": "结构名", "desc": "核心描述（几颗世界元、什么形状、怎么受力）"}],
  "new_characters": [{"name": "角色名", "role": "角色定位（如新学员、同期对手、高年级等）", "desc": "一段话描述：外貌+第一次出场时的印象+关键特征"}],
  "physical_details": [
    {
      "character": "角色名",
      "item": "物品/装备名（如魔能体、手套、带子）",
      "appearance": "外貌/外观描述",
      "chapter": 当前章节号
    }
  ],
  "equipment_updates": {
    "角色名": {"glove": "手套等级", "魔能体": "魔能体型号", "其他装备名": "描述"}
  },
  "grade_info": {
    "characters": ["角色名(年级)", "角色名(年级)"]
  }
}
不确定的字段填空字符串，列表留空数组。

foreshadowing_resolved 规则：prompt 中已给出"当前待揭示伏笔"列表。逐一检查本章正文是否明确揭开了其中某条伏笔的内容（剧情给出了结果或答案）。如果确认已揭示，将该条伏笔的 item（从待揭示列表中复制）和 keywords 放入 foreshadowing_resolved。不要自行编造不存在的伏笔。如果本章没有揭示任何伏笔，foreshadowing_resolved 留空数组。

角色命名规则：如果某个角色之前用占位名（如「中等身材学员」），本章中已揭示正式姓名（如「沈鸣」），在 characters 中只输出正式姓名，不要同时输出占位名。

new_characters 规则：只输出本章中首次出现正式姓名的配角（有名有姓+第一次出场），不输出占位名（如「高个子女生」「中年男人」）。不输出已在角色.md中登记过的角色。"""

COMPRESSOR_SYSTEM = """你是一个小说摘要压缩器。将本卷所有章节摘要压缩为500字以内的卷级叙事总结。
保留核心情节推进、关键转折和情感弧线。去掉次要细节。"""

RECALL_PLANNER_PROMPT = """你是一个小说写作规划助手。
动笔前制定一个小计划：判断这章要写什么，以及写之前需要回忆哪些已知事实。

以JSON格式输出，只输出JSON：
{
  "plan": "本章的写作计划（一句话概括推进方向）",
  "need_confirm": [
    {"type": "equipment", "characters": ["角色名1"]},
    {"type": "fact", "query": "需要确认的具体事实描述"},
    {"type": "scene", "query": "需要检索的场景描述"}
  ],
  "vector_queries": ["语义搜索词1"]
}

规则：
- type=equipment 用于确认角色装备，只填角色名
- type=fact 用于确认某个事件是否已发生、某个结论是否已成立
- type=scene 用于检索场景描写
- vector_queries 是直接去原文搜索的关键词
- 不需要的项留空数组
- 所有内容必须和已发生的情节一致，不能编造"""

CONFIRMATION_PROMPT = """你是一个事实确认助手。
判断检索到的原文片段是否明确回答了查询问题。

查询：{query}
检索到的原文片段：
{snippets}

以JSON格式输出：
{"found": true/false, "conclusion": "确切的结论（引用原文依据）", "chapter": 章节号}

规则：
- 只有片段中明确提到该事实时，found 才为 true
- 不要推测，不要脑补
- 如果找不到明确依据，found 为 false，conclusion 填空字符串"""


def estimate_tokens(text: str) -> int:
    return count_tokens(text)


def _build_knowledge_sections() -> list[str]:
    parts = []
    world = read_knowledge_file("世界观.md")
    if world:
        parts.append("【世界观与规则】\n" + world)
    chars = read_knowledge_file("角色.md")
    if chars:
        parts.append("【角色设定】\n" + chars)
    style = read_knowledge_file("风格.md")
    if style:
        parts.append("【写作风格】\n" + style)
    rules = read_knowledge_file("续写规范.md")
    if rules:
        parts.append("【续写规范】\n" + rules)
    return parts


_VOL_NUMS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


def _get_current_volume_outline() -> str:
    full = read_knowledge_file("大纲.md")
    if not full:
        return ""
    state = load_state()
    vol_num = state.get("current_volume", {}).get("num", 1)
    vol_cn = _VOL_NUMS[vol_num] if vol_num < len(_VOL_NUMS) else str(vol_num)
    sections = full.split("\n## ")
    current_idx = None
    next_idx = None
    for i, s in enumerate(sections):
        heading = s.split("\n")[0]
        if re.search(rf"第{vol_cn}卷", heading):
            current_idx = i
            continue
        if next_idx is None and vol_num < 5:
            next_cn = _VOL_NUMS[vol_num + 1]
            if re.search(rf"第{next_cn}卷", heading):
                next_idx = i
    result_parts = []
    if current_idx is not None:
        result_parts.append("## " + sections[current_idx])
    if next_idx is not None:
        next_vol = "## " + sections[next_idx]
        next_lines = next_vol.split("\n")[:8]
        next_vol = "\n".join(next_lines)
        result_parts.append("\n\n## 下一卷前瞻\n" + next_vol)
    result = "\n\n".join(result_parts)
    if len(result) > 3500:
        result = result[:3500]
    return result


def build_writer_prompt(state: dict) -> str:
    parts = []
    parts.append(WRITER_PERSONA)
    parts.extend(_build_knowledge_sections())
    outline = _get_current_volume_outline()
    if outline:
        parts.append("【当前卷大纲】\n" + outline)

    disc_summary = get_discussion_summary(max_chars=500)
    if disc_summary:
        parts.append(disc_summary)
    summary_text = get_summary_text(current_volume_only=False)
    if summary_text:
        parts.append(summary_text)
    word_control = WRITER_WORD_CONTROL.format(
        word_min=config.CHAPTER_WORD_COUNT_MIN,
        word_max=config.CHAPTER_WORD_COUNT_MAX,
    )
    parts.append(word_control)
    return "\n\n".join(parts)


def build_writer_human(state: dict, user_direction: str = "",
                       rewrite_reason: str = "",
                       recent_chapters_text: str = "",
                       recall_snippets: list[dict] = None,
                       writing_plan: str = "",
                       confirmed_facts: str = "",
                       rewrite_history: list[str] = None,
                       chapter_num: int = None) -> str:
    parts = []

    if rewrite_history:
        lines = [f"第{i+1}次打回：{r}" for i, r in enumerate(rewrite_history)]
        parts.append("【重写历史】\n" + "\n".join(lines))

    if state.get("current_scene"):
        parts.append(f"📍 当前场景：{state['current_scene']}")
    if state.get("current_time"):
        parts.append(f"⏰ 当前时间：{state['current_time']}")
    chars = state.get("characters", {})
    if chars:
        char_lines = []
        for name, info in chars.items():
            loc = info.get("location", "?")
            st = info.get("state", "")
            eq = info.get("equipment", {})
            eq_str = ", ".join(f"{k}={v}" for k, v in eq.items()) if eq else ""
            line = f"  · {name}：{loc}"
            if st:
                line += f"（{st})"
            if eq_str:
                line += f" [{eq_str}]"
            char_lines.append(line)
        parts.append("📍 角色位置：\n" + "\n".join(char_lines))
    if state.get("current_suspense"):
        parts.append(f"🎯 当前悬念：{state['current_suspense']}")

    if writing_plan:
        parts.append(f"【写作计划】\n{writing_plan}")

    if confirmed_facts:
        parts.append(f"【确认的事实】\n{confirmed_facts}")

    foreshadowing = state.get("foreshadowing", {})
    pending = [f for f in foreshadowing.get("pending", []) if f.get("importance", 3) >= 3]
    if pending:
        lines = [f"  · {f['item']}" for f in pending[:5]]
        parts.append("⏳ 活跃伏笔：\n" + "\n".join(lines))

    scenes = state.get("scene_catalog", [])
    at_risk = [s for s in scenes if "建议暂停" in s.get("risk", "")]
    if at_risk:
        risk_lines = [f"  ⚠️ {s['type']}（{s['risk']}）" for s in at_risk[:3]]
        parts.append("场景类型提醒：\n" + "\n".join(risk_lines))

    tense_values = state.get("_recent_tension", [])
    if len(tense_values) >= 3 and all(t >= config.RHYTHM_CONFIG["high_tension_threshold"] for t in tense_values[-3:]):
        parts.append(f"⚠️ 叙事节拍：最近{len(tense_values)}章张力值{tense_values}，"
                     f"连续高张力，建议考虑低张力过渡。")

    if recall_snippets and config.VECTOR_MEMORY_ENABLED:
        recall_lines = []
        seen_chapters = set()
        for r in recall_snippets:
            ch = r["chapter"]
            ch_label = f"第{ch}章"
            recall_lines.append(f"  [{ch_label}] {r['text'][:300]}")
            seen_chapters.add(ch)
        if recall_lines:
            parts.append("【记忆检索 — 相关原文片段】\n" + "\n\n".join(recall_lines))

    if recent_chapters_text:
        parts.append(f"【最近章节原文】\n{recent_chapters_text}")

    if rewrite_reason:
        parts.append(f"【打回重写原因】\n{rewrite_reason}")
    if user_direction:
        parts.append(f"【作者方向】\n{user_direction}")

    ch_label = chapter_num if chapter_num is not None else state['next_chapter']
    parts.append(f"\n请直接写第{ch_label}章正文，{config.CHAPTER_WORD_COUNT_MIN}-{config.CHAPTER_WORD_COUNT_MAX}字。")

    return "\n\n".join(parts)


def build_discussion_prompt(state: dict, recall_snippets: list[dict] = None) -> str:
    parts = []
    parts.append(DISCUSSION_PERSONA)
    parts.extend(_build_knowledge_sections())
    outline = _get_current_volume_outline()
    if outline:
        parts.append("【当前卷大纲】\n" + outline)
    disc_summary = get_discussion_summary(max_chars=500)
    if disc_summary:
        parts.append(disc_summary)
    summary_text = get_summary_text(current_volume_only=False)
    if summary_text:
        parts.append(summary_text)
    if recall_snippets and config.VECTOR_MEMORY_ENABLED:
        recall_lines = []
        for r in recall_snippets:
            recall_lines.append(f"  [第{r['chapter']}章] {r['text'][:300]}")
        parts.append("【检索到的相关原文片段】\n" + "\n".join(recall_lines))
    char_info_parts = []
    for name, info in state.get("characters", {}).items():
        loc = info.get("location", "?")
        st = info.get("state", "")
        bk = info.get("behavior_kernel", {})
        if bk:
            kernel_str = "; ".join(f"{k}={v}" for k, v in bk.items() if k != "last_updated_chapter")
            char_info_parts.append(f"  · {name}：{loc}（{st}) kernel:[{kernel_str}]")
        else:
            char_info_parts.append(f"  · {name}：{loc}（{st})")
    if char_info_parts:
        parts.append("【角色状态】\n" + "\n".join(char_info_parts))
    vol = state.get("current_volume", {})
    parts.append(f"当前进度：第{state['next_chapter']}章 · 第{vol.get('num', 1)}卷「{vol.get('title', '')}」")
    parts.append("【重要：讨论与写作是隔离的】写作时的小说家只能看到知识文件，看不到这段对话。"
                 "因此讨论中达成的任何共识——包括选择了哪个方案、确认了哪种写法、修正了什么理解——都必须写入知识文件。"
                 "【区分设定与剧情】世界观.md、角色.md 等知识文件中的描述是背景设定，不是已发生的叙事。你只能引用标注了「第X章：」的摘要内容来描述已发生的剧情，不得将设定描述当作正文引用来使用。"
                 "当作者向你提出创意想法时：分析影响 → 提供多个方案 → 指出风险 → 给出推荐。"
                 "你的回复末尾必须输出JSON的modification_plan，options中每个option包含files_to_modify列表。"
                 "files_to_modify每个元素：file（文件名）、action、以及对应参数。文件列表：世界观.md、角色.md、大纲.md、风格.md、续写规范.md。"
                 "支持的action类型："
                 "  · append_section — 文件末尾新增节，需：section_title（节标题）、content（内容）"
                 "  · replace_section — 替换指定节，需：section_title（匹配现有节标题）、content（新内容，含标题行）。优先用此操作而非全文替换"
                 "  · replace_content — 精确查找替换，需：old_text（要替换的原文本）、new_text（新文本）。用于改一句话/一段描述"
                 "  · replace_file — 全文替换（慎用），需：content（完整新内容）"
                 "文件选择规则：选择了开篇方案/章节写法/表达修正 → 续写规范.md；"
                 "修改了世界观规则/新增设定 → 世界观.md；调整了角色性格/背景 → 角色.md；"
                 "讨论了写作风格 → 风格.md；改了故事走向/里程碑 → 大纲.md。"
                 "【关于大纲.md的章节方向管理】各卷末尾有一个 ### ⚡ 当前推进方向 — 动态区域 节——"
                 "那是你为下一章写的具体方向。每次讨论完本章怎么写后，用 replace_section（section_title=\"⚡ 当前推进方向 — 动态区域\"）更新这个节："
                 "先写已完成章节的进展，再写下一章的战术方向（具体场景、谁出场、避免什么、不要省略什么）。"
                 "写作时的小说家会优先遵循这个方向。写完后，下轮讨论如果章节已完成，再次更新它。"
                 "注意：当前推进方向 — 动态区域 节一定是各卷的最后一个节，替换时新内容末尾要保留 --- 分隔线，然后接下一卷的标题。")
    return "\n\n".join(parts)


def build_extractor_prompt(chapter_text: str, state: dict) -> tuple[str, str]:
    prev_time = state.get("current_time", "")
    human = f"请分析第{state['next_chapter']}章。\n\n"
    if prev_time:
        human += f"上一章时间：{prev_time}\n\n"
    pending_fs = state.get("foreshadowing", {}).get("pending", [])
    if pending_fs:
        lines = [f"  · {f['item']}" for f in pending_fs[:15]]
        human += "当前待揭示伏笔：\n" + "\n".join(lines) + "\n\n"
    human += f"正文：\n{chapter_text[:6000]}"
    return EXTRACTOR_SYSTEM, human


def build_compressor_prompt(summaries_text: str) -> tuple[str, str]:
    human = f"请将以下章节摘要压缩为500字以内卷级总结：\n\n{summaries_text}"
    return COMPRESSOR_SYSTEM, human
