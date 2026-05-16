import json
import os
import re
import glob
from typing import Optional
import config


DEFAULT_STATE = {
    "next_chapter": 1,
    "novel_title": "",
    "current_volume": {"num": 1, "title": "第一卷"},
    "current_scene": "",
    "current_time": "第1卷第1天·开始",
    "characters": {},
    "foreshadowing": {
        "pending": [],
        "resolved": [],
    },
    "suspense": [],
    "current_suspense": "",
    "timeline_events": [],
    "volume_history": [],
    "self_calibrations": [],
    "health_report": {},
    "scene_catalog": [],
    "章节悬念类型历史": [],
    "_direction_chapter": 0,
}


def _state_path() -> str:
    return os.path.join(config.get_novel_knowledge_dir(), "state.json")


def _validate_state(state: dict) -> dict:
    if not isinstance(state.get("next_chapter"), int) or state["next_chapter"] < 1:
        state["next_chapter"] = DEFAULT_STATE["next_chapter"]
    if not isinstance(state.get("characters"), dict):
        state["characters"] = {}
    if not isinstance(state.get("foreshadowing"), dict):
        state["foreshadowing"] = {"pending": [], "resolved": []}
    else:
        if "pending" not in state["foreshadowing"]:
            state["foreshadowing"]["pending"] = []
        if "resolved" not in state["foreshadowing"]:
            state["foreshadowing"]["resolved"] = []
    if not isinstance(state.get("current_volume"), dict):
        state["current_volume"] = dict(DEFAULT_STATE["current_volume"])
    if "_direction_chapter" not in state:
        state["_direction_chapter"] = 0
    return state


def _summaries_path() -> str:
    return os.path.join(config.get_novel_knowledge_dir(), "summaries.json")


def _discussion_log_path() -> str:
    return os.path.join(config.get_novel_knowledge_dir(), "discussion_log.json")


def _snapshots_dir() -> str:
    return os.path.join(config.get_novel_knowledge_dir(), "snapshots")


def _events_path() -> str:
    return os.path.join(config.get_novel_knowledge_dir(), "snapshots", "events.jsonl")


def _style_progression_path() -> str:
    return os.path.join(config.get_novel_knowledge_dir(), "style_progression.json")


def _physical_details_path() -> str:
    return os.path.join(config.get_novel_knowledge_dir(), "physical_details.json")


def load_state() -> dict:
    path = _state_path()
    if not os.path.exists(path):
        return dict(DEFAULT_STATE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _validate_state({**DEFAULT_STATE, **data})
    except (json.JSONDecodeError, FileNotFoundError):
        return dict(DEFAULT_STATE)


def save_state(state: dict):
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_summaries() -> list[dict]:
    path = _summaries_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_summaries(summaries: list[dict]):
    path = _summaries_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


def append_summary(chapter: int, summary: str):
    summaries = load_summaries()
    existing = [s for s in summaries if s["chapter"] == chapter]
    if existing:
        existing[0]["summary"] = summary
    else:
        summaries.append({"chapter": chapter, "summary": summary})
    save_summaries(summaries)


def get_summary_text(current_volume_only: bool = True) -> str:
    state = load_state()
    summaries = load_summaries()
    if current_volume_only and state["volume_history"]:
        last_vol = state["volume_history"][-1]
        start_ch = int(last_vol["chapters"].split("-")[0]) if "-" in last_vol["chapters"] else 1
    else:
        start_ch = 1
    lines = []
    for s in summaries:
        if s["chapter"] >= start_ch:
            lines.append(f"第{s['chapter']}章：{s['summary']}")
    vol_lines = []
    for v in state.get("volume_history", []):
        vol_lines.append(f"第{v['num']}卷「{v['title']}」({v['chapters']})：{v['summary']}")
    result = ""
    if vol_lines:
        result += "【旧卷摘要】\n" + "\n".join(vol_lines) + "\n\n"
    if lines:
        result += "【逐章摘要】\n" + "\n".join(lines)
    return result.strip()


def compress_to_volume_summary() -> str:
    return ""


def load_discussion_log() -> dict:
    path = _discussion_log_path()
    if not os.path.exists(path):
        return {"sessions": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"sessions": []}


def save_discussion_log(log: dict):
    path = _discussion_log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def get_discussion_summary(max_chars: int = 500) -> str:
    log = load_discussion_log()
    sessions = log.get("sessions", [])
    if not sessions:
        return ""
    parts = ["【讨论决策摘要】"]
    for s in sessions[-5:]:
        summary = s.get("summary", "")
        decisions = s.get("decisions", [])
        if summary:
            parts.append(f"  · {summary}")
        for d in decisions:
            file = d.get("file", "")
            section = d.get("section", "") or d.get("change", "")
            parts.append(f"    - {file}：{section}" if file else f"    - {section}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def save_snapshot(chapter_num: int, state: dict, summaries: list):
    snap_dir = _snapshots_dir()
    os.makedirs(snap_dir, exist_ok=True)
    snap = {"chapter": chapter_num, "state": state, "summaries": summaries}
    path = os.path.join(snap_dir, f"ch_{chapter_num:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)


def list_snapshots() -> list[int]:
    snap_dir = _snapshots_dir()
    if not os.path.isdir(snap_dir):
        return []
    result = []
    for fn in sorted(os.listdir(snap_dir), reverse=True):
        m = re.match(r"ch_(\d+)\.json", fn)
        if m:
            result.append(int(m.group(1)))
    return result


def rollback_to_chapter(chapter_num: int, rollback_discussions: bool = False) -> dict:
    snap_dir = _snapshots_dir()
    path = os.path.join(snap_dir, f"ch_{chapter_num:03d}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"快照不存在：第{chapter_num}章")

    with open(path, "r", encoding="utf-8") as f:
        snap = json.load(f)

    save_state(snap["state"])
    save_summaries(snap["summaries"])

    for f in glob.glob(os.path.join(config.get_novel_output_dir(), "第*章*.md")):
        bn = os.path.basename(f)
        m = re.match(r"第(\d+)章", bn)
        if m and int(m.group(1)) > chapter_num:
            os.remove(f)

    for f in glob.glob(os.path.join(snap_dir, "ch_*.json")):
        bn = os.path.basename(f)
        m = re.match(r"ch_(\d+)", bn)
        if m and int(m.group(1)) > chapter_num:
            os.remove(f)

    disc_pre_pat = re.compile(r"disc_(\d+)_pre\.json")
    disc_post_pat = re.compile(r"disc_(\d+)_post\.json")
    for f in sorted(os.listdir(snap_dir)):
        if disc_pre_pat.match(f) or disc_post_pat.match(f):
            from_json = os.path.join(snap_dir, f)
            try:
                if rollback_discussions and disc_pre_pat.match(f):
                    with open(from_json, "r", encoding="utf-8") as df:
                        disc_snap = json.load(df)
                    for kf, content in disc_snap.get("files", {}).items():
                        write_knowledge_file(kf, content)
                os.remove(from_json)
            except Exception:
                pass

    record_event("rollback", {"target_chapter": chapter_num, "rollback_discussions": rollback_discussions})

    if config.VECTOR_MEMORY_ENABLED:
        try:
            from novel_memory import NovelMemory
            memory = NovelMemory(config.ACTIVE_NOVEL)
            count = memory.rebuild_index()
        except Exception as e:
            print(f"⚠️ [rollback vector sync] 向量库重建失败: {e}")

    return snap["state"]


def save_discussion_snapshot_pre(disc_id: str, knowledge_files: list[str]):
    snap_dir = _snapshots_dir()
    os.makedirs(snap_dir, exist_ok=True)
    files = {}
    for fname in knowledge_files:
        content = read_knowledge_file(fname)
        files[fname] = content
    snap = {"disc_id": disc_id, "files": files}
    path = os.path.join(snap_dir, f"{disc_id}_pre.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)


def save_discussion_snapshot_post(disc_id: str):
    snap_dir = _snapshots_dir()
    os.makedirs(snap_dir, exist_ok=True)
    state = load_state()
    path = os.path.join(snap_dir, f"{disc_id}_post.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"disc_id": disc_id, "state": state}, f, ensure_ascii=False, indent=2)


def record_event(event_type: str, data: dict):
    path = _events_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {"type": event_type, **data}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_timeline_events(since_chapter: int = 0) -> list[dict]:
    path = _events_path()
    if not os.path.exists(path):
        return []
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                ch = event.get("chapter", 0) or 0
                if ch >= since_chapter or event["type"] in ("discussion", "rollback"):
                    result.append(event)
            except json.JSONDecodeError:
                continue
    return result


def read_knowledge_file(filename: str) -> str:
    path = os.path.join(config.get_novel_knowledge_dir(), filename)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def write_knowledge_file(filename: str, content: str):
    path = os.path.join(config.get_novel_knowledge_dir(), filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, path)


def save_output_chapter(chapter_num: int, chapter_text: str) -> str:
    out_dir = config.get_novel_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    filename = f"第{chapter_num:03d}章.md"
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(chapter_text)
    return path


def delete_output_chapters_after(chapter_num: int):
    out_dir = config.get_novel_output_dir()
    if not os.path.isdir(out_dir):
        return
    for f in glob.glob(os.path.join(out_dir, "第*章*.md")):
        bn = os.path.basename(f)
        m = re.match(r"第(\d+)章", bn)
        if m and int(m.group(1)) > chapter_num:
            os.remove(f)


def load_style_progression() -> dict:
    path = _style_progression_path()
    if not os.path.exists(path):
        return {"volumes": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"volumes": []}


def save_style_progression(data: dict):
    path = _style_progression_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_physical_details() -> list[dict]:
    path = _physical_details_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_physical_details(details: list[dict]):
    path = _physical_details_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)


def get_physical_details_for_character(name: str) -> list[dict]:
    all_details = load_physical_details()
    return [d for d in all_details if d.get("character", "").strip() == name.strip()]


def append_physical_details(new_details: list[dict]):
    details = load_physical_details()
    existing_keys = set(
        (d.get("character", "").strip(), d.get("item", "").strip())
        for d in details
    )
    added = 0
    for nd in new_details:
        key = (nd.get("character", "").strip(), nd.get("item", "").strip())
        if key not in existing_keys and key[0] and key[1]:
            details.append(nd)
            existing_keys.add(key)
            added += 1
    if added > 0:
        save_physical_details(details)
    return added
