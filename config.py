import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

NOVELIST_CONFIG = {
    "model": os.getenv("NOVELIST_MODEL", "deepseek-v4-pro"),
    "temperature": float(os.getenv("NOVELIST_TEMPERATURE", "1.0")),
    "max_tokens": int(os.getenv("NOVELIST_MAX_TOKENS", "6000")),
    "top_p": float(os.getenv("NOVELIST_TOP_P", "1.0")),
    "frequency_penalty": float(os.getenv("NOVELIST_FREQUENCY_PENALTY", "0.3")),
    "presence_penalty": float(os.getenv("NOVELIST_PRESENCE_PENALTY", "0.2")),
    "enable_thinking": os.getenv("NOVELIST_ENABLE_THINKING", "false").lower() in ("true", "1", "yes"),
    "reasoning_effort": os.getenv("NOVELIST_REASONING_EFFORT", "medium"),
}

EXTRACTOR_CONFIG = {
    "model": os.getenv("EXTRACTOR_MODEL", "deepseek-v4-flash"),
    "temperature": float(os.getenv("EXTRACTOR_TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("EXTRACTOR_MAX_TOKENS", "2000")),
    "top_p": float(os.getenv("EXTRACTOR_TOP_P", "1.0")),
    "frequency_penalty": float(os.getenv("EXTRACTOR_FREQUENCY_PENALTY", "0")),
    "presence_penalty": float(os.getenv("EXTRACTOR_PRESENCE_PENALTY", "0")),
    "enable_thinking": os.getenv("EXTRACTOR_ENABLE_THINKING", "false").lower() in ("true", "1", "yes"),
    "reasoning_effort": os.getenv("EXTRACTOR_REASONING_EFFORT", "low"),
    "response_format": {"type": "json_object"},
}

COMPRESSOR_CONFIG = {
    "model": os.getenv("COMPRESSOR_MODEL", "deepseek-v4-flash"),
    "temperature": float(os.getenv("COMPRESSOR_TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("COMPRESSOR_MAX_TOKENS", "2000")),
    "top_p": float(os.getenv("COMPRESSOR_TOP_P", "1.0")),
    "frequency_penalty": float(os.getenv("COMPRESSOR_FREQUENCY_PENALTY", "0")),
    "presence_penalty": float(os.getenv("COMPRESSOR_PRESENCE_PENALTY", "0")),
    "enable_thinking": os.getenv("COMPRESSOR_ENABLE_THINKING", "false").lower() in ("true", "1", "yes"),
    "reasoning_effort": os.getenv("COMPRESSOR_REASONING_EFFORT", "low"),
}

CHAPTER_WORD_COUNT_MIN = int(os.getenv("CHAPTER_WORD_COUNT_MIN", "2200"))
CHAPTER_WORD_COUNT_MAX = int(os.getenv("CHAPTER_WORD_COUNT_MAX", "2800"))
CONTEXT_SOFT_LIMIT_TOKENS = int(os.getenv("CONTEXT_SOFT_LIMIT_TOKENS", "30000"))
RECENT_CHAPTERS_COUNT = int(os.getenv("RECENT_CHAPTERS_COUNT", "2"))

RHYTHM_CONFIG = {
    "max_consecutive_high_tension": int(os.getenv("RHYTHM_MAX_CONSECUTIVE_HIGH_TENSION", "4")),
    "high_tension_threshold": int(os.getenv("RHYTHM_HIGH_TENSION_THRESHOLD", "7")),
    "max_consecutive_same_scene_type": int(os.getenv("RHYTHM_MAX_CONSECUTIVE_SAME_SCENE_TYPE", "3")),
}

ACTIVE_NOVEL = os.getenv("NOVELNEXUS_ACTIVE_NOVEL", "default")
NOVELS_DIR = os.path.join(BASE_DIR, "novels")

# ── Vector Memory (V4新增) ──
VECTOR_MEMORY_ENABLED = os.getenv("VECTOR_MEMORY_ENABLED", "true").lower() in ("true", "1", "yes")
VECTOR_MEMORY_MODEL = os.getenv("VECTOR_MEMORY_MODEL", "BAAI/bge-small-zh-v1.5")
VECTOR_MEMORY_SEARCH_K = int(os.getenv("VECTOR_MEMORY_SEARCH_K", "5"))
VECTOR_MEMORY_RECENCY_BOOST = float(os.getenv("VECTOR_MEMORY_RECENCY_BOOST", "0.3"))

PRICING = {
    "deepseek-v4-flash": {
        "input_cache_hit": 0.2,
        "input_cache_miss": 2.0,
        "output": 3.0,
    },
    "deepseek-v4-pro": {
        "input_cache_hit": 0.2,
        "input_cache_miss": 2.0,
        "output": 3.0,
    },
    "deepseek-chat": {
        "input_cache_hit": 0.2,
        "input_cache_miss": 2.0,
        "output": 3.0,
    },
    "deepseek-reasoner": {
        "input_cache_hit": 0.2,
        "input_cache_miss": 2.0,
        "output": 3.0,
    },
}


def get_novel_dir(novel_name: str = None) -> str:
    name = novel_name or ACTIVE_NOVEL
    return os.path.join(NOVELS_DIR, name)


def get_novel_knowledge_dir(novel_name: str = None) -> str:
    return os.path.join(get_novel_dir(novel_name), "knowledge")


def get_novel_output_dir(novel_name: str = None) -> str:
    return os.path.join(get_novel_dir(novel_name), "output")


def set_active_novel(name: str) -> None:
    global ACTIVE_NOVEL
    ACTIVE_NOVEL = name


def list_novels() -> list[str]:
    if not os.path.isdir(NOVELS_DIR):
        return []
    return [d for d in os.listdir(NOVELS_DIR)
            if os.path.isdir(os.path.join(NOVELS_DIR, d)) and not d.startswith(".")]
