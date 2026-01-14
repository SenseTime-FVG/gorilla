import json
import os
import time
from types import SimpleNamespace
from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from overrides import override



class DeepSeekAPIHandler(OpenAICompletionsHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        base = "https://api.deepseek.com"

        self.client = OpenAI(
            base_url=os.getenv("DEEPSEEK_BASE_URL", base),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        
        # 添加额外的参数
        self.extra_body = kwargs

    # The deepseek API is unstable at the moment, and will frequently give empty responses, so retry on JSONDecodeError is necessary
    @retry_with_backoff(error_type=[RateLimitError, json.JSONDecodeError], error_message_pattern=r".*Insufficient Balance.*")
    def generate_with_backoff(self, **kwargs):
        """
        Per the DeepSeek API documentation:
        https://api-docs.deepseek.com/quick_start/rate_limit

        DeepSeek API does NOT constrain user's rate limit. We will try out best to serve every request.
        But please note that when our servers are under high traffic pressure, you may receive 429 (Rate Limit Reached) or 503 (Server Overloaded). When this happens, please wait for a while and retry.

        Thus, backoff is still useful for handling 429 and 503 errors.
        """
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        
        # 添加给 lightllm v1 接口测试
        extra_info = {"extra_body": self.extra_body}
        
        # 检测 message，如果太长的话就直接抛出异常
        try:
            # 安全检查：确保 message 不为空
            if not message:
                raise ValueError("Message list is empty")
            
            # 安全检查：获取最后一个消息的 content
            last_message = message[-1]
            if "content" not in last_message:
                raise ValueError("Last message does not have 'content' field")
            
            last_message_content = last_message["content"]
            # 安全检查：确保 content 是字符串类型
            if not isinstance(last_message_content, str):
                raise ValueError(f"Last message content is not a string, got {type(last_message_content)}")
            
            # 检查长度
            if len(last_message_content) > 500000:
                # 尝试找到第一个 user 消息作为 user_query
                user_query = ""
                for msg in message:
                    if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                        user_query = msg["content"]
                        break
                if not user_query:
                    user_query = message[0].get("content", "") if message else ""
                    if isinstance(user_query, str):
                        user_query = user_query
                    else:
                        user_query = str(user_query)
                
                raise ValueError(f"DeepSeek API Handler: last message content too long ({len(last_message_content)} chars). User query: {user_query}")
        except (ValueError, KeyError, IndexError, TypeError) as e:
            # 只捕获预期的异常类型，避免捕获 KeyboardInterrupt, SystemExit 等
            # 创建一个模拟的 API 响应对象，包含后续解析所需的所有字段
            # 将异常信息放到 content 中，方便后续查看
            error_content = f"Error during message validation, skipped API call. Error: {str(e)}"
            print(f"DeepSeek API Handler error: {error_content}")
            fake_message = SimpleNamespace(
                content=error_content,
                tool_calls=None,
            )
            fake_choice = SimpleNamespace(message=fake_message)
            fake_usage = SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
            )
            fake_api_response = SimpleNamespace(
                choices=[fake_choice],
                usage=fake_usage,
            )
            return fake_api_response, 0.0

        if len(tools) > 0:
            return self.generate_with_backoff(
                model=self.model_name,
                messages=message,
                tools=tools,
                temperature=self.temperature,
                **extra_info
            )
        else:
            return self.generate_with_backoff(
                model=self.model_name,
                messages=message,
                temperature=self.temperature,
                **extra_info
            )

    @override
    def _query_prompting(self, inference_data: dict):
        """
        This method is intended to be used by the `DeepSeek-R1` models. If used for other models, you will need to modify the code accordingly.

        Reasoning models don't support temperature parameter
        https://api-docs.deepseek.com/guides/reasoning_model

        `DeepSeek-R1` should use `deepseek-reasoner` as the model name in the API
        https://api-docs.deepseek.com/quick_start/pricing
        """
        message: list[dict] = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(message)}

        return self.generate_with_backoff(
            model=self.model_name,
            messages=message,
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        # 'deepseek-reasoner does not support successive user messages, so we need to combine them
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        return {"message": []}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        DeepSeek does not take reasoning content in next turn chat history, for both prompting and function calling mode.
        Error: Error code: 400 - {'error': {'message': 'The reasoning_content is an intermediate result for display purposes only and will not be included in the context for inference. Please remove the reasoning_content from your message to reduce network traffic.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
        """
        response_data = super()._parse_query_response_prompting(api_response)
        self._add_reasoning_content_if_available_prompting(api_response, response_data)
        return response_data

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        DeepSeek does not take reasoning content in next turn chat history, for both prompting and function calling mode.
        Error: Error code: 400 - {'error': {'message': 'The reasoning_content is an intermediate result for display purposes only and will not be included in the context for inference. Please remove the reasoning_content from your message to reduce network traffic.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
        """
        response_data = super()._parse_query_response_FC(api_response)
        self._add_reasoning_content_if_available_FC(api_response, response_data)
        return response_data
