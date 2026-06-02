#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一模型调用接口
支持 Ollama 本地模型和 OpenAI-Compatible API 云端模型
"""

import os
import time
import requests
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class ModelCaller:
    """
    统一的大模型调用封装类
    
    支持的 model_name:
      本地 Ollama:
        - 'qwen2.5:1.5b'
        - 'deepseek-r1:1.5b'
        - 'qwen2.5-coder:1.5b'
        - 'smollm:1.7b'
        - 'gemma2:2b'
        - 'qwen3-vl:2b'
      云端 API:
        - 'deepseek-v4-pro'
        - 'kimi-k2.6'
        - 'minimax-m2.7'
    """

    MODEL_CONFIGS = {
        # ===== Ollama 本地模型 =====
        # 超微 <1B
        'qwen3.5:0.8b': {'type': 'ollama', 'ctx': 32768},
        # 微型 1-2B
        'qwen2.5:1.5b': {'type': 'ollama', 'ctx': 32768},
        'deepseek-r1:1.5b': {'type': 'ollama', 'ctx': 131072},
        'qwen2.5-coder:1.5b': {'type': 'ollama', 'ctx': 32768},
        'smollm:1.7b': {'type': 'ollama', 'ctx': 8192},
        'gemma2:2b': {'type': 'ollama', 'ctx': 8192},
        'qwen3-vl:2b': {'type': 'ollama', 'ctx': 32768},
        'glm-4.7-flash:q8_0': {'type': 'ollama', 'ctx': 32768},
        # 小型 3B（规模梯度）
        'qwen2.5:3b': {'type': 'ollama', 'ctx': 32768},
        'llama3.2:3b': {'type': 'ollama', 'ctx': 131072},
        # 中型 4-9B（规模梯度补全）
        'phi4-mini:3.8b': {'type': 'ollama', 'ctx': 16384},
        'deepseek-r1:7b': {'type': 'ollama', 'ctx': 131072},
        'llama3.1:8b': {'type': 'ollama', 'ctx': 131072},
        'qwen3.5:9b': {'type': 'ollama', 'ctx': 32768},
        # 大型 14B
        'qwen2.5:14b': {'type': 'ollama', 'ctx': 32768},
        'deepseek-r1:14b': {'type': 'ollama', 'ctx': 131072},
        # 超大 27-35B（本地上界参照）
        'qwen3.5:27b': {'type': 'ollama', 'ctx': 32768},
        'qwen3.6:35b': {'type': 'ollama', 'ctx': 32768},
        'qwen3.5:35b': {'type': 'ollama', 'ctx': 32768},
        # 云端 API 模型
        'deepseek-v4-pro': {
            'type': 'api',
            'base_url': 'https://api.deepseek.com',
            'model_id': 'deepseek-v4-pro',
            'env_key': 'DEEPSEEK_API_KEY',
            'input_price': 1.74,
            'output_price': 3.48,
        },
        'kimi-k2.6': {
            'type': 'api',
            'base_url': 'https://api.moonshot.cn/v1',
            'model_id': 'kimi-k2.6',
            'env_key': 'KIMI_API_KEY',
            'input_price': 0.95,
            'output_price': 4.00,
        },
        'minimax-m2.7': {
            'type': 'api',
            'base_url': 'https://api.minimax.chat/v1',
            'model_id': 'minimax-m2.7',
            'env_key': 'MINIMAX_API_KEY',
            'input_price': 0.30,
            'output_price': 1.20,
        },
    }

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self._api_clients: Dict[str, OpenAI] = {}

    def _get_api_client(self, config: Dict[str, Any]) -> OpenAI:
        """获取或创建 API 客户端（带缓存）"""
        base_url = config['base_url']
        if base_url not in self._api_clients:
            api_key = os.getenv(config['env_key'])
            if not api_key:
                raise ValueError(f"未设置环境变量: {config['env_key']}")
            self._api_clients[base_url] = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        return self._api_clients[base_url]

    def call(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        timeout: int = 300,
        rag_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        统一调用接口
        
        Args:
            model_name: 模型名称，需在 MODEL_CONFIGS 中
            prompt: 用户输入的 prompt
            temperature: 采样温度（0-1）
            system_prompt: 系统提示词（可选）
            max_tokens: 最大生成 token 数
            timeout: 请求超时秒数
        
        Returns:
            dict: {
                'content': str,          # 模型生成的文本
                'input_tokens': int,     # 输入 token 数
                'output_tokens': int,    # 输出 token 数
                'total_tokens': int,     # 总 token 数
                'latency_ms': float,     # 请求耗时（毫秒）
                'cost_usd': float,       # 预估成本（美元，API 模型有效）
            }
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"不支持的模型: {model_name}. "
                f"支持的模型: {list(self.MODEL_CONFIGS.keys())}"
            )

        config = self.MODEL_CONFIGS[model_name]
        # 若提供 RAG 上下文，拼接到 prompt 前
        if rag_context and rag_context.strip():
            prompt = f"{rag_context.strip()}\n\n---\n\n{prompt}"
        start_time = time.time()

        if config['type'] == 'ollama':
            result = self._call_ollama(
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            result = self._call_api(
                config=config,
                prompt=prompt,
                temperature=temperature,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        latency_ms = (time.time() - start_time) * 1000
        result['latency_ms'] = round(latency_ms, 2)
        return result

    def _call_ollama(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        system_prompt: Optional[str],
        max_tokens: int,
        timeout: int,
    ) -> Dict[str, Any]:
        """调用 Ollama 本地服务"""
        # qwen3/qwq 系列内部推理会消耗大量 token，需禁用显式思考并扩大预算
        is_thinking_model = any(
            model_name.startswith(prefix)
            for prefix in ("qwen3", "qwq")
        )
        effective_max_tokens = max(max_tokens, 2048) if is_thinking_model else max_tokens

        options: Dict[str, Any] = {
            'temperature': temperature,
            'num_ctx': self.MODEL_CONFIGS[model_name].get('ctx', 8192),
            'num_predict': effective_max_tokens,
        }
        if is_thinking_model:
            options['think'] = False

        payload = {
            'model': model_name,
            'prompt': prompt,
            'stream': False,
            'options': options,
        }
        if system_prompt:
            payload['system'] = system_prompt

        response = requests.post(
            f"{self.ollama_host}/api/generate",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()

        input_tokens = data.get('prompt_eval_count', 0)
        output_tokens = data.get('eval_count', 0)

        return {
            'content': data.get('response', ''),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost_usd': 0.0,  # 本地模型成本为 0
        }

    def _call_api(
        self,
        config: Dict[str, Any],
        prompt: str,
        temperature: float,
        system_prompt: Optional[str],
        max_tokens: int,
        timeout: int,
    ) -> Dict[str, Any]:
        """调用云端 API"""
        client = self._get_api_client(config)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=config['model_id'],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        msg = resp.choices[0].message
        # 兼容 DeepSeek/Kimi 等推理模型：content 可能为空，答案在 reasoning_content
        content = msg.content or ""
        if not content:
            content = getattr(msg, "reasoning_content", "") or ""
        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # 估算成本
        input_cost = (input_tokens / 1e6) * config['input_price']
        output_cost = (output_tokens / 1e6) * config['output_price']
        total_cost = input_cost + output_cost

        return {
            'content': content,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost_usd': round(total_cost, 6),
        }

    def list_available_models(self) -> list:
        """返回当前可用的模型列表（含本地和云端）"""
        available = []
        for name, cfg in self.MODEL_CONFIGS.items():
            if cfg['type'] == 'ollama':
                available.append(name)
            else:
                # 检查环境变量是否存在
                if os.getenv(cfg['env_key']):
                    available.append(name)
        return available


def smoke_test():
    """冒烟测试：验证所有可用模型"""
    print("=" * 60)
    print("ModelCaller 冒烟测试")
    print("=" * 60)

    caller = ModelCaller()
    models = caller.list_available_models()
    print(f"\n检测到可用模型: {models}\n")

    test_prompt = "请用一句话介绍自己。"

    for model in models:
        print(f"\n--- 测试模型: {model} ---")
        try:
            result = caller.call(
                model_name=model,
                prompt=test_prompt,
                temperature=0.3,
                max_tokens=100,
            )
            content = result['content'].strip().replace('\n', ' ')
            if len(content) > 80:
                content = content[:80] + "..."
            print(f"[OK] 延迟={result['latency_ms']}ms | Token={result['total_tokens']} | 成本=${result['cost_usd']}")
            print(f"     回复: {content}")
        except Exception as e:
            print(f"[FAIL] {e}")

    print("\n" + "=" * 60)
    print("冒烟测试完成")
    print("=" * 60)


if __name__ == "__main__":
    smoke_test()
