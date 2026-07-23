# about habitat scene
from __future__ import annotations
import os

INVALID_SCENE_ID = []

API_MODE = "qwen" # "gpt" or "qwen"

Qwen_END_POINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
Qwen_OPENAI_KEY = os.environ.get("QWEN_API_KEY", "YOUR_QWEN_API_KEY")

GPT_END_POINT = "YOUR_GPT_ENDPOINT"
GPT_OPENAI_KEY = "YOUR_GPT_API_KEY"