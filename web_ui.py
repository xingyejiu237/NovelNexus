import gradio as gr
import os
import json
import sys
import io

# 强制 UTF-8 输出，防止 emoji 在 GBK 终端崩溃
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import config
from state_manager import (
    load_state, list_snapshots, rollback_to_chapter,
    read_knowledge_file, write_knowledge_file, get_summary_text,
    get_timeline_events,
)
from novelist import generate_chapter, accept_chapter, discuss, execute_modifications, init_novel
from token_tracker import tracker

_mode = "writing"
_chapter_text = ""
_agent_suggestion = ""
_usage_info = {}
_volume_switch = None
_last_result = None
_rewrite_history = []
_discussion_history = []
_pending_modifications = []


def _calc_writing_time(total_chars: int) -> str:
    if total_chars <= 0:
        return "暂无数据"
    return f"约{total_chars * 0.001 * 6:.0f}小时"


def _format_token_display():
    summary = tracker.get_summary()
    novel = config.ACTIVE_NOVEL
    novel_path = config.get_novel_dir()
    total_chars = 0
    if os.path.isdir(config.get_novel_output_dir()):
        for fn in os.listdir(config.get_novel_output_dir()):
            if fn.endswith(".md"):
                fpath = os.path.join(config.get_novel_output_dir(), fn)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                        total_chars += len(text)
                except Exception:
                    pass
    total_cost = summary["total_cost_yuan"]
    total_call = summary["call_count"]
    cache_hit = summary["total_cache_hit"]
    input_total = summary["total_input_only"]
    cache_pct = f" ({cache_hit / input_total * 100:.0f}%)" if input_total > 0 and cache_hit > 0 else ""
    cache_str = f" | ♻️ {cache_hit:,}t{cache_pct}" if cache_hit > 0 else " | ♻️ 0"
    write_time = _calc_writing_time(total_chars)

    memory_str = ""
    if config.VECTOR_MEMORY_ENABLED:
        try:
            from novel_memory import NovelMemory
            mem = NovelMemory(config.ACTIVE_NOVEL, lazy=True)
            if mem.has_index():
                stats = mem.get_stats()
                memory_str = f" | 🧠 {stats['total_chunks']}段"
        except Exception:
            pass

    return f"📊 {novel} | {total_chars}字 ({write_time}) | {total_call}次 | ¥{total_cost:.4f}{cache_str}{memory_str}"


def on_set_novel(novel_name):
    global _chapter_text, _agent_suggestion, _usage_info, _volume_switch, _last_result, _discussion_history
    _chapter_text = ""
    _agent_suggestion = ""
    _usage_info = {}
    _volume_switch = None
    _last_result = None
    _rewrite_history = []
    _discussion_history = []
    config.set_active_novel(novel_name)
    state = load_state()
    ch = state["next_chapter"]
    vol = state.get("current_volume", {})
    status = f"📖 第{ch}章 · 第{vol.get('num',1)}卷「{vol.get('title','')}」 | 🏠 {state.get('current_scene','?')}"
    token_disp = _format_token_display()
    novels = config.list_novels()
    dropdown_choices = novels if novels else ["default"]
    current_val = novel_name if novel_name in dropdown_choices else dropdown_choices[0]
    return "", "(待生成)", status, token_disp, gr.Dropdown(choices=dropdown_choices, value=current_val)


def on_new_novel(name, idea, protagonist):
    if not name.strip():
        return "⚠️ 请填写小说名", None
    result = init_novel(name, idea, protagonist)
    if result["success"]:
        novels = config.list_novels()
        current_val = name if name in novels else (novels[0] if novels else "default")
        return f"✅ 小说《{name}》已创建", gr.Dropdown(choices=novels, value=current_val)
    return f"❌ {result.get('message', '创建失败')}", None


def on_generate(user_direction, rewrite_checkbox, rewrite_reason, recall_input, chapter_display, suggestion_display, status_display, token_display, mode_state):
    global _chapter_text, _agent_suggestion, _usage_info, _volume_switch, _last_result, _rewrite_history

    state = load_state()
    ch = state["next_chapter"] - 1 if rewrite_checkbox else state["next_chapter"]
    ch = max(1, ch)
    yield chapter_display, suggestion_display, f"⏳ 正在生成第{ch}章...", token_display, mode_state

    recall_queries = None
    if recall_input and recall_input.strip():
        queries = [q.strip() for q in recall_input.strip().split(";") if q.strip()]
        if queries:
            recall_queries = queries

    _rewrite_history = _rewrite_history or []
    if rewrite_checkbox and rewrite_reason.strip():
        _rewrite_history.append(rewrite_reason.strip())

    result = generate_chapter(
        user_direction=user_direction,
        is_rewrite=rewrite_checkbox,
        rewrite_reason=rewrite_reason or "",
        recall_queries=recall_queries,
        rewrite_history=_rewrite_history,
    )

    if "error" in result:
        yield chapter_display, suggestion_display, f"❌ {result['error']}", token_display, mode_state
        return

    _last_result = result
    _chapter_text = result["chapter_text"]
    _agent_suggestion = result["agent_suggestion"]
    _usage_info = result["usage_info"]
    _volume_switch = result.get("volume_switch")

    chapter_display = f"## 第{result['chapter_num']}章\n\n{_chapter_text}"
    alerts = result.get("alerts", [])
    alerts_block = ""
    if alerts:
        alerts_block = "### ⚠️ 告警\n\n" + "\n".join(f"- ⚠️ {a}" for a in alerts) + "\n\n---\n\n"
    writer_questions = result.get("writer_questions", "")
    questions_block = ""
    if writer_questions:
        questions_block = f"> ❓ **写手提问**：\n\n{writer_questions}\n\n---\n\n"
    title_review = result.get("title_review")
    title_note = ""
    if title_review and title_review.get("rewritten"):
        title_note = f"✏️ 标题已修正：「{title_review['original_title']}」→「{title_review['new_title']}」"
    conflict_alert = result.get("conflict_alert", "")
    if conflict_alert:
        suggestion_display = f"### ⚠️ 一致性警示\n\n{conflict_alert}"
        if alerts_block:
            suggestion_display = alerts_block + suggestion_display
        if title_note:
            suggestion_display = f"### ✏️ 标题修正\n\n{title_note}\n\n---\n\n{suggestion_display}"
        if questions_block:
            suggestion_display += f"\n\n{questions_block}"
        if _agent_suggestion:
            suggestion_display += f"\n\n---\n### 💬 Agent附言\n\n{_agent_suggestion}"
    elif title_note:
        suggestion_display = f"### ✏️ 标题修正\n\n{title_note}"
        if alerts_block:
            suggestion_display = alerts_block + suggestion_display
        if questions_block:
            suggestion_display += f"\n\n{questions_block}"
        if _agent_suggestion:
            suggestion_display += f"\n\n---\n### 💬 Agent附言\n\n{_agent_suggestion}"
    else:
        suggestion_display = alerts_block or ""
        if questions_block:
            if suggestion_display:
                suggestion_display += f"\n\n{questions_block}"
            else:
                suggestion_display = questions_block
        if _agent_suggestion:
            if suggestion_display:
                suggestion_display += f"\n\n---\n### 💬 Agent附言\n\n{_agent_suggestion}"
            else:
                suggestion_display = _agent_suggestion
        if not suggestion_display:
            suggestion_display = "(无附言)"
    status = ""
    if rewrite_checkbox:
        status = f"🔄 已重写第{result['chapter_num']}章"
    token_display = _format_token_display()
    yield chapter_display, suggestion_display, status, token_display, mode_state


def on_accept(chapter_display, suggestion_display, status_display, token_display, rollback_dropdown, mode_state):
    global _chapter_text, _agent_suggestion, _usage_info, _volume_switch, _last_result

    if not _chapter_text or not _last_result:
        chs = list_snapshots()
        chs_str = [str(c) for c in chs] if chs else ["（暂无快照）"]
        val = chs_str[0] if chs_str else "（暂无快照）"
        yield chapter_display, suggestion_display, "⚠️ 没有可接受的章节", token_display, gr.Dropdown(choices=chs_str, value=val), mode_state
        return

    ch = _last_result["chapter_num"]
    result = accept_chapter(_chapter_text, _agent_suggestion, _usage_info, _volume_switch, chapter_num=ch)

    status_display = f"✅ 第{ch}章已保存！"
    token_display = _format_token_display()

    _chapter_text = ""
    _agent_suggestion = ""
    _usage_info = {}
    _volume_switch = None
    _last_result = None
    _rewrite_history = []

    chs = list_snapshots()
    chs_str = [str(c) for c in chs] if chs else ["（暂无快照）"]
    val = chs_str[0] if chs_str else "（暂无快照）"
    yield "", "(待生成)", status_display, token_display, gr.Dropdown(choices=chs_str, value=val), mode_state


def _render_value(value, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(value, str):
        return value
    elif isinstance(value, (int, float, bool)):
        return str(value)
    elif isinstance(value, dict):
        if not value:
            return ""
        lines = []
        for k, v in value.items():
            rendered = _render_value(v, indent + 1)
            if rendered:
                if "\n" in rendered:
                    lines.append(f"{prefix}**{k}**：")
                    lines.append(rendered)
                else:
                    lines.append(f"{prefix}**{k}**：{rendered}")
        return "\n".join(lines)
    elif isinstance(value, list):
        if not value:
            return ""
        lines = []
        for item in value:
            rendered = _render_value(item, indent)
            if rendered:
                lines.append(f"{prefix}- {rendered}")
        return "\n".join(lines)
    return ""


def _format_file_modifications(files_to_modify: list) -> str:
    if not files_to_modify:
        return ""
    lines = []
    for fm in files_to_modify:
        fname = fm.get("file", "?")
        action = fm.get("action", "append")
        title = fm.get("section_title", "")
        content = fm.get("content", "")
        action_label = "🆕 新增" if action in ("append_section", "append") else "✏️ 替换"
        lines.append(f"📄 `{fname}` {action_label}")
        if title:
            lines.append(f"   └ 节：{title}")
        if content:
            preview = content[:200].replace("\n", " ")
            lines.append(f"   └ 内容：{preview}...")
    return "\n".join(lines)


def _format_plan_markdown(plan: dict) -> str:
    lines = []

    has_impact = bool(plan.get("impact_analysis") or plan.get("analysis"))
    consumed_keys = {"impact_analysis", "analysis", "recommended_options", "options", "plans", "recommendation", "suggestion", "next_step"}

    if has_impact:
        impact = plan.get("impact_analysis") or plan.get("analysis") or {}
        lines.append("### 📊 影响分析")
        rendered = _render_value(impact)
        if rendered:
            lines.append(rendered)
        lines.append("")

    options = (plan.get("recommended_options")
               or plan.get("options")
               or plan.get("plans")
               or [])
    if options:
        has_any_file_mod = any(
            isinstance(opt, dict) and opt.get("files_to_modify")
            for opt in options
        )
        if has_any_file_mod:
            lines.append("> ⚠️ 以下回复包含 **文件修改建议**。点击「✅ 确认执行修改」按钮才会真正写入文件。")
            lines.append("")

        lines.append("### 📋 推荐方案")
        for i, opt in enumerate(options, 1):
            if isinstance(opt, str):
                lines.append(f"**方案{i}**：{opt}")
            elif isinstance(opt, dict):
                name = opt.get("name") or opt.get("label") or opt.get("title", f"方案{i}")
                lines.append(f"**方案{i}：{name}**")

                files_mod = opt.pop("files_to_modify", None)

                for k, v in opt.items():
                    if k in ("name", "label", "title"):
                        continue
                    rendered = _render_value(v)
                    if rendered:
                        lines.append(rendered)

                if files_mod:
                    lines.append("")
                    lines.append(_format_file_modifications(files_mod))

                if files_mod is not None:
                    opt["files_to_modify"] = files_mod
            lines.append("")

    recommendation = plan.get("recommendation") or plan.get("suggestion") or ""
    if recommendation:
        lines.append("> 💡 " + _render_value(recommendation))

    next_step = plan.get("next_step", "")
    if next_step:
        lines.append(f"\n📌 {_render_value(next_step)}")

    leftover = {k: v for k, v in plan.items() if k not in consumed_keys
                and not k.startswith("_") and v}
    if leftover:
        if lines:
            lines.append("---")
        for k, v in leftover.items():
            rendered = _render_value(v)
            if rendered:
                lines.append(f"**{k}**：{rendered}")

    return "\n".join(lines)


def _format_modifications_panel(mods: list) -> str:
    if not mods:
        return "*暂无待执行的修改*"
    lines = ["## 📋 待执行的修改", ""]
    for i, mod in enumerate(mods, 1):
        fname = mod.get("file", "?")
        action = mod.get("action", "append_section")
        if action == "append_section":
            action_label = "🆕 新增节"
            detail = mod.get("section_title", "")
        elif action == "replace_section":
            action_label = "✏️ 替换节"
            detail = mod.get("section_title", "")
        elif action == "replace_content":
            action_label = "🔧 精确替换"
            detail = mod.get("old_text", "")[:50]
        elif action == "replace_file":
            action_label = "📝 全文替换"
            detail = ""
        else:
            action_label = action
            detail = ""
        lines.append(f"### {i}. 📄 `{fname}` — {action_label}")
        if detail:
            lines.append(f"**目标**：{detail}")
        content = mod.get("content", mod.get("new_text", ""))
        if content:
            preview = content[:500].strip()
            lines.append("")
            lines.append("```markdown")
            lines.append(preview)
            if len(content) > 500:
                lines.append("...")
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def on_discuss(user_message, chat_history, mod_panel):
    global _last_result, _pending_modifications
    if not user_message.strip():
        return chat_history, user_message, mod_panel
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    result = discuss(user_message, chat_history=chat_history)
    if "error" in result:
        chat_history.append({"role": "assistant", "content": f"❌ {result['error']}"})
        return chat_history, "", mod_panel
    reply = result.get("reply", "（无回复）")
    plan = result.get("modification_plan")
    new_mod_panel = mod_panel
    if plan:
        options = (plan.get("recommended_options")
                   or plan.get("options")
                   or plan.get("plans")
                   or [])
        if options:
            recommended = options[0]
            files_to_modify = recommended.get("files_to_modify", [])
            modifications = []
            for fm in files_to_modify:
                mod_entry = {
                    "file": fm.get("file", ""),
                    "action": fm.get("action", "append_section"),
                }
                if mod_entry["action"] in ("replace_content",):
                    mod_entry["old_text"] = fm.get("old_text", "")
                    mod_entry["new_text"] = fm.get("new_text", fm.get("content", fm.get("changes", "")))
                elif mod_entry["action"] == "replace_file":
                    mod_entry["content"] = fm.get("content", fm.get("changes", ""))
                else:
                    mod_entry["section_title"] = fm.get("section_title", "")
                    mod_entry["content"] = fm.get("content", fm.get("changes", ""))
                modifications.append(mod_entry)
            _pending_modifications = modifications
            new_mod_panel = _format_modifications_panel(modifications)
            reply += f"\n\n> 📋 已提取修改方案，共 {len(modifications)} 项，请在下方面板确认后点击「✅ 确认执行修改」"
    _last_result = result
    chat_history.append({"role": "assistant", "content": reply})
    return chat_history, "", new_mod_panel


def on_confirm_modifications(chat_history, mod_panel):
    global _pending_modifications
    if not _pending_modifications:
        if chat_history:
            chat_history.append({"role": "assistant", "content": "⚠️ 没有待确认的修改方案"})
        return chat_history, mod_panel
    result = execute_modifications(_pending_modifications)
    _pending_modifications = []
    if result["success"]:
        msg = f"✅ {result['message']}"
    else:
        msg = f"❌ {result.get('message', '执行失败')}"
    if chat_history:
        chat_history.append({"role": "assistant", "content": msg})
    return chat_history, _format_modifications_panel([])


def on_reject_modifications(chat_history, mod_panel):
    global _pending_modifications
    _pending_modifications = []
    if chat_history:
        chat_history.append({"role": "assistant", "content": "🗑️ 已放弃本次修改方案"})
    return chat_history, _format_modifications_panel([])


def on_rollback(chapter_str, rollback_discussions):
    global _chapter_text, _agent_suggestion, _usage_info, _volume_switch, _last_result, _rewrite_history
    _chapter_text = ""
    _agent_suggestion = ""
    _usage_info = {}
    _volume_switch = None
    _last_result = None
    _rewrite_history = []
    if not chapter_str or chapter_str == "（暂无快照）":
        chs = list_snapshots()
        chs_str = [str(c) for c in chs] if chs else ["（暂无快照）"]
        return "", "(待生成)", "⚠️ 请先选择要回退到的章节", _format_token_display(), gr.Dropdown(choices=chs_str, value=chs_str[0] if chs_str else "（暂无快照）")
    try:
        ch = int(chapter_str)
    except (ValueError, TypeError):
        chs = list_snapshots()
        chs_str = [str(c) for c in chs] if chs else ["（暂无快照）"]
        return "", "(待生成)", "⚠️ 无效章节号", _format_token_display(), gr.Dropdown(choices=chs_str, value=chs_str[0] if chs_str else "（暂无快照）")
    try:
        state = rollback_to_chapter(ch, rollback_discussions)
    except Exception as e:
        chs = list_snapshots()
        chs_str = [str(c) for c in chs] if chs else ["（暂无快照）"]
        return "", "(待生成)", f"❌ 回退失败：{e}", _format_token_display(), gr.Dropdown(choices=chs_str, value=chs_str[0] if chs_str else "（暂无快照）")
    chs = list_snapshots()
    chs_str = [str(c) for c in chs] if chs else ["（暂无快照）"]
    next_ch = state["next_chapter"]
    status_msg = f"🔄 已回退到第{ch}章，下一章为第{next_ch}章"
    direction_ch = state.get("_direction_chapter", 0)
    if direction_ch and direction_ch < next_ch:
        status_msg += (
            f"\n\n⚠️ 当前方向建议计划在第{direction_ch}章附近，但回退后下一页为第{next_ch}章。"
            f"请检查「我的方向建议.md」是否需要更新当前章节计划。"
        )
    return "", "(待生成)", status_msg, _format_token_display(), gr.Dropdown(choices=chs_str, value=chs_str[0] if chs_str else "（暂无快照）")


def on_refresh_rollback():
    chs = list_snapshots()
    chs_str = [str(c) for c in chs] if chs else ["（暂无快照）"]
    return gr.Dropdown(choices=chs_str, value=chs_str[0] if chs_str else "（暂无快照）")


def on_reload_state(knowledge_html, status_display, token_display):
    state = load_state()
    ch = state["next_chapter"]
    vol = state.get("current_volume", {})
    status = f"📖 第{ch}章 · 第{vol.get('num',1)}卷「{vol.get('title','')}」 | 🏠 {state.get('current_scene','?')}"
    token = _format_token_display()
    knowledge = _build_knowledge_panel()
    return knowledge, status, token


def switch_to_writing(mode_state, chat_history):
    global _mode, _discussion_history
    _mode = "writing"
    return "writing", chat_history or []


def switch_to_discussion(mode_state, chat_history):
    global _mode
    _mode = "discussion"
    return "discussion", chat_history or []


def on_undo_last():
    chs = list_snapshots()
    if len(chs) >= 2:
        return on_rollback(str(chs[1]), False)
    return "", "(待生成)", "⚠️ 没有更早的快照可回退", _format_token_display(), gr.Dropdown()

CSS = """
.container { max-width: 1400px; margin: 0 auto; }
.knowledge-panel { font-size: 14px; }
.knowledge-panel h3 { margin-top: 8px; margin-bottom: 4px; }
.chapter-box { border-left: 3px solid #6b5bff; padding-left: 16px; }
"""


def _build_knowledge_panel():
    state = load_state()
    lines = [f"📚 **{state.get('novel_title', '未命名')}**"]
    vol = state.get("current_volume", {})
    lines.append(f"📖 第{state['next_chapter']}章 · 第{vol.get('num',1)}卷「{vol.get('title','')}」")
    lines.append(f"🏠 {state.get('current_scene', '?')} | ⏰ {state.get('current_time', '?')}")
    if state.get("current_suspense"):
        lines.append(f"🎯 {state['current_suspense']}")
    lines.append("")
    for fname in ["世界观.md", "角色.md", "大纲.md", "风格.md", "续写规范.md"]:
        content = read_knowledge_file(fname)
        if content and len(content) > 3:
            lines.append(f"📄 **{fname}** ({len(content)}字)")
            lines.append(f"<details><summary>展开</summary>\n\n```\n{content[:800]}{'...' if len(content) > 800 else ''}\n```\n</details>")
        else:
            lines.append(f"📄 **{fname}** (空)")
    return "\n\n".join(lines)


with gr.Blocks(title="NovelNexus V3 - 小说家") as demo:
    gr.Markdown("# 📖 NovelNexus V3\n**你的专属小说家** — 讨论剧情 · 创作章节 · 持续积累")

    state = load_state()
    novels_list = config.list_novels()
    current_novel = config.ACTIVE_NOVEL if config.ACTIVE_NOVEL in novels_list else (novels_list[0] if novels_list else "default")

    mode_state = gr.State("writing")
    mode_state.value = "writing"

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📚 小说")
            novel_dropdown = gr.Dropdown(
                choices=novels_list if novels_list else ["default"],
                value=current_novel,
                label="切换小说",
                interactive=True,
            )
            with gr.Accordion("✨ 新小说", open=False):
                new_novel_name = gr.Textbox(label="小说名", placeholder="新小说名")
                new_novel_idea = gr.Textbox(label="核心创意", placeholder="你想写什么样的故事？", lines=2)
                new_novel_protagonist = gr.Textbox(label="主角", placeholder="主角名字 + 一句话人设")
                create_novel_btn = gr.Button("创建")
                create_novel_msg = gr.Markdown("")

            panel_refresh_btn = gr.Button("🔄 刷新知识面板", size="sm")

            knowledge_display = gr.Markdown(
                value=_build_knowledge_panel(),
                elem_classes="knowledge-panel",
            )

            with gr.Accordion("⏪ 回退", open=False):
                rollback_dropdown = gr.Dropdown(
                    choices=[str(c) for c in list_snapshots()] or ["（暂无快照）"],
                    value=str((list_snapshots() or ["（暂无快照）"])[0]) if list_snapshots() else "（暂无快照）",
                    label="快照章节",
                    interactive=True,
                )
                with gr.Row():
                    rollback_refresh_btn = gr.Button("🔄 刷新", size="sm", scale=1)
                    undo_btn = gr.Button("↩️ 撤销上一章", size="sm", scale=1)
                rollback_btn = gr.Button("⏪ 回退到选定章节", variant="stop", size="sm")
                rollback_disc_cb = gr.Checkbox(label="同时撤销中间讨论", value=False)

        with gr.Column(scale=2, min_width=520):
            with gr.Tabs() as tabs:
                with gr.TabItem("✍️ 写作", id="writing_tab"):
                    chapter_display = gr.Markdown(
                        value="*点击「生成」开始写第一章*" if state["next_chapter"] == 1 else "*续写模式*",
                        elem_classes="chapter-box",
                    )
                    suggestion_display = gr.Textbox(label="💬 审查与反馈", value="(待生成)", interactive=False, lines=4)

                    with gr.Row():
                        generate_btn = gr.Button("🚀 生成", variant="primary", scale=2)
                        accept_btn = gr.Button("✅ 接受并保存", variant="secondary", scale=1)
                        undo_shortcut_btn = gr.Button("↩️ 撤销", size="sm", scale=1)

                    user_input = gr.Textbox(
                        label="✍️ 你的方向（可选）",
                        placeholder="留空 = AI 自动续写。写方向：\"这章突出紧张感\" / 回答写手上一章提出的问题",
                        lines=2,
                    )
                    recall_input = gr.Textbox(
                        label="🔍 记忆检索（可选）",
                        placeholder="检索前文细节，用 ; 分隔多个关键词。如：苏棠的银色带子; 青藤廊道",
                        lines=1,
                    )
                    with gr.Row():
                        rewrite_checkbox = gr.Checkbox(label="🔄 打回重写", info="勾选后下方原因生效")
                        rewrite_reason = gr.Textbox(label="打回原因", placeholder="上一章哪里不对？", lines=1, scale=2)

                with gr.TabItem("💬 讨论", id="discussion_tab"):
                    chatbot = gr.Chatbot(label="💬 与小说家讨论剧情", height=300)
                    with gr.Row():
                        discuss_input = gr.Textbox(label="你的想法", placeholder="想加个设定/改个情节/讨论方向...", scale=3, lines=2)
                        discuss_btn = gr.Button("💬 讨论", variant="primary", scale=1)
                    mod_panel = gr.Markdown(value="*暂无待执行的修改*", label="📋 修改预览")
                    with gr.Row():
                        confirm_mod_btn = gr.Button("✅ 确认执行修改", variant="secondary")
                        reject_mod_btn = gr.Button("🗑️ 放弃修改", variant="stop", size="sm")

            status_display = gr.Markdown(
                value=f"📖 第{state['next_chapter']}章 · 第{state.get('current_volume',{}).get('num',1)}卷「{state.get('current_volume',{}).get('title','')}」 | 🏠 {state.get('current_scene', '?')}"
            )
            token_display = gr.Markdown(value=_format_token_display())

            # Writing mode buttons
            generate_btn.click(
                fn=on_generate,
                inputs=[user_input, rewrite_checkbox, rewrite_reason, recall_input, chapter_display, suggestion_display, status_display, token_display, mode_state],
                outputs=[chapter_display, suggestion_display, status_display, token_display, mode_state],
            )

            accept_btn.click(
                fn=on_accept,
                inputs=[chapter_display, suggestion_display, status_display, token_display, rollback_dropdown, mode_state],
                outputs=[chapter_display, suggestion_display, status_display, token_display, rollback_dropdown, mode_state],
            )

            undo_shortcut_btn.click(
                fn=on_undo_last,
                inputs=[],
                outputs=[chapter_display, suggestion_display, status_display, token_display, rollback_dropdown],
            )

            # Discussion mode buttons
            discuss_btn.click(
                fn=on_discuss,
                inputs=[discuss_input, chatbot, mod_panel],
                outputs=[chatbot, discuss_input, mod_panel],
            ).then(
                fn=lambda: "",
                outputs=[discuss_input],
            )

            confirm_mod_btn.click(
                fn=on_confirm_modifications,
                inputs=[chatbot, mod_panel],
                outputs=[chatbot, mod_panel],
            )

            reject_mod_btn.click(
                fn=on_reject_modifications,
                inputs=[chatbot, mod_panel],
                outputs=[chatbot, mod_panel],
            )

            # Novel management
            novel_dropdown.change(
                fn=on_set_novel,
                inputs=[novel_dropdown],
                outputs=[chapter_display, suggestion_display, status_display, token_display, novel_dropdown],
            )

            create_novel_btn.click(
                fn=on_new_novel,
                inputs=[new_novel_name, new_novel_idea, new_novel_protagonist],
                outputs=[create_novel_msg, novel_dropdown],
            ).then(
                fn=lambda: (""),
                outputs=[new_novel_name],
            )

            panel_refresh_btn.click(
                fn=on_reload_state,
                inputs=[knowledge_display, status_display, token_display],
                outputs=[knowledge_display, status_display, token_display],
            )

            # Rollback
            rollback_refresh_btn.click(fn=on_refresh_rollback, inputs=[], outputs=[rollback_dropdown])
            rollback_btn.click(
                fn=on_rollback,
                inputs=[rollback_dropdown, rollback_disc_cb],
                outputs=[chapter_display, suggestion_display, status_display, token_display, rollback_dropdown],
            )


if __name__ == "__main__":
    print("🔄 NovelNexus V3 启动中...")
    import sys
    sys.stdout.flush()

    if config.VECTOR_MEMORY_ENABLED:
        import threading
        import sys as _sys

        def _init_memory():
            try:
                from novel_memory import NovelMemory
                mem = NovelMemory(lazy=True)
                if not mem.has_index():
                    count = mem.rebuild_index()
                    print(f"🗂️ 向量记忆：已从{count}章重建索引")
                else:
                    stats = mem.get_stats()
                    print(f"🗂️ 向量记忆：{stats['total_chunks']}段已索引")
                _sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ 向量记忆初始化失败: {e}")
                _sys.stdout.flush()

        t = threading.Thread(target=_init_memory, daemon=True)
        t.start()
        t.join(timeout=5)
        if t.is_alive():
            print("⏩ 向量记忆后台加载中（耗时>5s），Web UI 先行启动")
            _sys.stdout.flush()

    _port = 7860
    _max_port_attempts = 10
    for attempt in range(_max_port_attempts):
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            if s.connect_ex(('127.0.0.1', _port)) == 0:
                s.close()
                raise OSError(f"Port {_port} in use")
            s.close()
            demo.launch(server_name="127.0.0.1", server_port=_port, css=CSS)
            break
        except Exception:
            if attempt < _max_port_attempts - 1:
                _port += 1
            else:
                raise
