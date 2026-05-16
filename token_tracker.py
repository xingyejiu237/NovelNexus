import json
import os
from typing import Optional
import config


class Tracker:
    def __init__(self, history_file: str = None):
        self.history_file = history_file or os.path.join(
            config.BASE_DIR, "token_cost_history.json"
        )
        self._records = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self.history_file):
            return {"__global__": []}
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"__global__": []}

    def _save(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    def add_usage(self, agent_name: str, model: str,
                  input_tokens: int, output_tokens: int,
                  cache_hit_tokens: int, is_pro: bool,
                  novel_name: str = None) -> float:
        novel = novel_name or config.ACTIVE_NOVEL
        pricing = config.PRICING.get(model, config.PRICING["deepseek-v4-flash"])
        input_cost = (
            cache_hit_tokens * pricing["input_cache_hit"]
            + (input_tokens - cache_hit_tokens) * pricing["input_cache_miss"]
        ) / 1_000_000
        output_cost = output_tokens * pricing["output"] / 1_000_000
        total_cost = input_cost + output_cost

        record = {
            "agent": agent_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit": cache_hit_tokens,
            "cost_yuan": round(total_cost, 4),
            "novel": novel,
        }

        if novel not in self._records:
            self._records[novel] = []
        self._records[novel].append(record)
        self._records["__global__"].append(record)
        self._save()
        return total_cost

    def get_summary(self, novel_name: str = None) -> dict:
        if novel_name:
            records = self._records.get(novel_name, [])
        else:
            records = self._records.get("__global__", [])
        total_cost = sum(r["cost_yuan"] for r in records)
        total_input = sum(r["input_tokens"] for r in records)
        total_output = sum(r["output_tokens"] for r in records)
        total_cache_hit = sum(r["cache_hit"] for r in records)
        return {
            "total_cost_yuan": round(total_cost, 4),
            "call_count": len(records),
            "total_cache_hit": total_cache_hit,
            "total_input": total_input + total_output,
            "total_input_only": total_input,
        }

    def get_all_novels_summary(self) -> dict:
        result = {}
        for key in self._records:
            if key == "__global__":
                continue
            result[key] = self.get_summary(key)
        return result


tracker = Tracker()
