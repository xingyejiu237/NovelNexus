# NovelNexus

基于 DeepSeek + 向量记忆的长篇小说 AI 写作助手。

## 核心能力

- **多智能体编排** — 写手(Writer) / 提取器(Extractor) / 讨论员(Discussant) / 一致性审查(Checker)，各司其职
- **向量记忆** — ChromaDB + sentence-transformers 语义检索，每次写作自动召回相关原文片段
- **状态一致性** — 逐章结构化提取、快照与回退、伏笔追踪、装备与物理细节持久化
- **回忆规划层** — 写前用 LLM 规划检索需求，精确确认事实，避免连续性断裂
- **费用与缓存监控** — 实时 token 统计、缓存命中率追踪、DeepSeek 定价自动计算

## 快速开始

```bash
git clone https://github.com/yourname/NovelNexus.git
cd NovelNexus_V3

pip install -r requirements.txt

cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

python web_ui.py
```

浏览器打开 `http://127.0.0.1:7860`。

## 文件架构

```
NovelNexus_V3/
├── novelist.py           # 写作编排 + 向量检索 + 精确确认
├── system_builder.py     # Prompt 组装（writer/discussant/extractor）
├── state_manager.py      # 状态持久化 + 快照回退 + 校验
├── novel_memory.py       # ChromaDB 向量库 + 时序加权
├── token_counter.py      # DeepSeek 官方 tokenizer
├── token_tracker.py      # 调用计费 + 缓存追踪
├── consistency_checker.py# 一致性审查器
├── config.py             # 集中配置
├── web_ui.py             # Gradio 前端
└── deepseek_v3_tokenizer/# 官方 tokenizer 文件
```

## 技术栈

- **LLM**: DeepSeek (flash / pro)
- **框架**: LangChain + Gradio
- **向量库**: ChromaDB + sentence-transformers
- **运行**: Python 3.10+

## 数据流

```
生成章节 ─→ 提取器解析（摘要/角色/装备/伏笔/悬念）
            ├─ 更新 state.json
            ├─ 更新 summaries.json
            ├─ 写入 output/ 章节文件
            └─ 索引进向量库（逐段 embedding）

写新章前 ─→ 回忆规划层规划检索需求
            ├─ 向量库 semantic search
            ├─ summaries 关键词匹配
            └─ 原文全文搜索（三级下探）
```
