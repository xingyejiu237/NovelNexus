import os
import config

_TOKENIZER_CACHE = {}
_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek_v3_tokenizer")


def count_tokens(text: str) -> int:
    cache_key = _TOKENIZER_DIR
    if cache_key not in _TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER_CACHE[cache_key] = AutoTokenizer.from_pretrained(
                _TOKENIZER_DIR, trust_remote_code=True
            )
        except Exception as e:
            print(f"⚠️ [token_counter] 加载tokenizer失败: {e}")
            _TOKENIZER_CACHE[cache_key] = None

    tokenizer = _TOKENIZER_CACHE[cache_key]
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            print(f"⚠️ [token_counter] 编码失败: {e}")

    return int(len(text) * 1.5)
