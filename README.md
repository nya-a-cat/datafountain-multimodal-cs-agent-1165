# DataFountain 1165 Multimodal Customer Service Agent

用于 DataFountain 1165「具有多模态能力的客服智能体设计」赛题的本地实现仓库。核心目标是提供一个可运行的 `/chat` API，完成多模态理解、知识检索增强、多轮对话和幻觉抑制。

## 项目结构

- `code/baseline_api.py`：主服务，提供 `/health` 和 `/chat`。
- `code/build_knowledge.py`：从手册原文构建知识库切片。
- `code/extract_images.py`：解压或整理官方插图。
- `code/generate_submission.py`：批量调用 `/chat` 生成提交文件。
- `code/generate_rerank_submission.py`：基于重排策略生成提交结果。
- `code/offline_retrieval_eval.py`：离线检索评估。
- `code/validate_submission.py`：校验提交 CSV 格式。
- `data/`：官方知识、题目和插图数据，默认不纳入版本控制。
- `submissions/`、`outputs/`：生成结果和实验产物，默认不纳入版本控制。

## 环境准备

建议使用 Python 虚拟环境后安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

## 启动服务

先配置模型相关环境变量，再启动 API：

```bash
export SILICONFLOW_API_KEY=你的密钥
export KAFU_API_TOKEN=可选的服务鉴权令牌
python -m uvicorn code.baseline_api:app --host 0.0.0.0 --port 8000
```

如果你使用的是其他兼容服务，也可以通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`CHAT_MODEL`、`VLM_MODEL` 覆盖默认值。

## 生成提交文件

1. 准备官方知识数据和插图。
2. 构建知识库：

```bash
python code/build_knowledge.py
```

3. 生成提交 CSV：

```bash
python code/generate_submission.py \
  --question-file data/question_public.csv \
  --out-file submissions/submission.csv \
  --attach-images \
  --images-dir data/images \
  --knowledge-file data/knowledge_v2.jsonl
```

4. 校验提交文件：

```bash
python code/validate_submission.py \
  --question-file data/question_public.csv \
  --submission-file submissions/submission.csv
```

## 配置说明

- `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `SILICONFLOW_API_KEY` / `DASHSCOPE_API_KEY`：模型服务密钥，代码会优先读取这些环境变量。
- `OPENAI_BASE_URL`：兼容接口地址。
- `KAFU_API_TOKEN`：可选，启用后 `/chat` 会检查 `Authorization: Bearer ...`。
- `USE_QUERY_REWRITE`：设为 `1` 时开启检索改写。
- `RAG_TRACE_PATH`：可选，写出检索与回答 trace。

## 上传前检查

- 确认 `.venv/`、`data/`、`submissions/`、`outputs/` 仍然被忽略。
- 不要把真实 API key、cookie、token 或个人账号信息写进代码和文档。
- 如果新增文档需要公开，记得同时更新 `.gitignore`。