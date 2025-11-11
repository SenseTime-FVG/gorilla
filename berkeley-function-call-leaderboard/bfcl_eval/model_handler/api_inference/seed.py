# doubao_requests_handler.py
import os
import time
import json
import getpass
from typing import Any, Dict, Tuple

from openai import RateLimitError
import openai_proxy

from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI


class DoubaoRequestsHandler(BaseHandler):
    """
    与 APIRequestsHandler 同输入/输出，但底层通过 openai_proxy.GptProxy(channel_code="doubao")。
    兼容豆包响应结构，并对 code=10001 做 30s 休眠后再本地重试一次。
    """

    def __init__(self, model_name: str, temperature: float) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.channel_code = "doubao"
        self._user = getpass.getuser()
        # 如需要强制 FC，可开启：
        # self.is_fc_model = True

    # -------------------- 通用解码 --------------------
    def decode_ast(self, result, language=ReturnFormat.PYTHON, has_tool_call_tag=True):
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag=True):
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result, has_tool_call_tag)

    # -------------------- 发送请求（豆包通道） --------------------
    @retry_with_backoff(error_type=Exception)
    def generate_with_backoff(self, **kwargs):
        """
        返回: (response_json: dict, latency_seconds: float)
        仅 code==10001 会本地 sleep 后重试一次；其他业务错误码直接返回合成响应，不触发重试。
        """
        if not self.api_key:
            raise EnvironmentError("请设置环境变量 OPENAI_API_KEY")

        client = openai_proxy.GptProxy(api_key=self.api_key)

        # 裁剪 system 中 "# Tool" 之后内容
        new_messages = []
        for message in kwargs.get("messages", []):
            if message.get("role") == "system":
                content = message.get("content", "")
                message = {**message, "content": content.split("# Tool")[0].strip()}
            new_messages.append(message)

        call_kwargs = dict(
            messages=new_messages,
            model=kwargs.get("model", self.model_name),
            channel_code=self.channel_code,
            transaction_id=f"{self._user}-{int(time.time()*1000)}",
        )
        tools = kwargs.get("tools")

        def _do_call():
            try:
                if tools:
                    return client.generate(**call_kwargs, tools=tools)
                return client.generate(**call_kwargs)
            except TypeError:
                # 代理不支持 tools 参数
                return client.generate(**call_kwargs)

        def _synthetic_error(code: int, msg: str) -> dict:
            # OpenAI 兼容的最小响应，方便后续 normalize & parse
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"[ERROR code={code}] {msg}"
                    }
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }

        start = time.time()

        # 第一次请求
        try:
            rsp = _do_call()
        except Exception as e:
            # HTTP/网络层错误仍然交给装饰器重试（如需也不重试，可改为 return _synthetic_error(-1, str(e)), latency）
            raise Exception(f"doubao proxy call failed: {e}")

        if not getattr(rsp, "ok", False):
            text = getattr(rsp, "text", "")
            # 非 200 仍抛异常 -> 交给装饰器重试；若也不想重试，可改为直接 return _synthetic_error(-1, text)
            raise Exception(f"doubao proxy returned non-OK: {text}")

        data = rsp.json()

        # 业务态处理：仅 code==10001 重试一次
        if isinstance(data, dict) and "code" in data and data.get("code") != 10000:
            code = int(data.get("code"))
            msg = str(data.get("msg", ""))

            if code == 10001:
                sleep_s = int(os.getenv("BFCL_TRANSIENT_SLEEP", "30"))
                time.sleep(sleep_s)
                try:
                    rsp2 = _do_call()
                except Exception as e:
                    # 二次调用失败，直接返回合成错误，不触发装饰器重试
                    end = time.time()
                    return _synthetic_error(code, f"{msg} | after-sleep call failed: {e}"), (end - start)

                if not getattr(rsp2, "ok", False):
                    end = time.time()
                    text2 = getattr(rsp2, "text", "")
                    return _synthetic_error(code, f"{msg} | after-sleep non-OK: {text2}"), (end - start)

                data2 = rsp2.json()
                if isinstance(data2, dict) and data2.get("code") != 10000:
                    end = time.time()
                    return _synthetic_error(int(data2.get("code")), str(data2.get("msg", ""))), (end - start)

                end = time.time()
                return data2, (end - start)

            # 其它任何业务错误码：直接返回合成响应（不抛异常，不重试）
            end = time.time()
            return _synthetic_error(code, msg), (end - start)

        end = time.time()
        return data, (end - start)

    # -------------------- 响应归一化 --------------------
    def _normalize_openai_like_response(self, api_response: Any) -> Tuple[Dict, Dict]:
        """
        统一返回 (message_obj, usage_obj)：
          - OpenAI: {"choices":[{"message": {...}}], "usage": {...}}
          - 豆包：{"code":10000,"msg":"成功","data":{"response_content":{ "choices":[...], "usage": {...}}}}
          - 展开包装层：data/result/response/output
          - 顶层文本/消息对象兜底
        """
        obj = api_response
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                return {"role": "assistant", "content": obj}, {}

        if not isinstance(obj, dict):
            return {"role": "assistant", "content": str(obj)}, {}

        # 展开一层常见 wrapper
        for wrapper in ("data", "result", "response", "output"):
            if isinstance(obj.get(wrapper), dict):
                obj = obj.get(wrapper)

        # 标准 OpenAI
        if "choices" in obj and obj.get("choices"):
            msg = obj["choices"][0].get("message") or {}
            usage = obj.get("usage", {})
            return msg, usage

        # 豆包 response_content
        if "response_content" in obj and isinstance(obj["response_content"], dict):
            rc = obj["response_content"]
            if "choices" in rc and rc.get("choices"):
                msg = rc["choices"][0].get("message") or {}
                usage = rc.get("usage", {}) or obj.get("usage", {})
                return msg, usage
            if "message" in rc and isinstance(rc["message"], dict):
                return rc["message"], rc.get("usage", {}) or obj.get("usage", {})
            for key in ("content", "text", "output_text"):
                if key in rc:
                    return {"role": "assistant", "content": rc[key]}, rc.get("usage", {}) or obj.get("usage", {})
            return {"role": "assistant", "content": str(rc)}, rc.get("usage", {}) or obj.get("usage", {})

        # 顶层文本 / 消息对象
        for key in ("content", "text", "output_text", "message"):
            if key in obj:
                val = obj[key]
                usage = obj.get("usage", {})
                if isinstance(val, str):
                    return {"role": "assistant", "content": val}, usage
                if isinstance(val, dict):
                    return val, usage
                return {"role": "assistant", "content": str(val)}, usage

        # 兜底报错（可读）
        raise KeyError(f"'choices' not found. keys={list(obj.keys())}, raw={str(api_response)[:500]}")

    # ==================== FC methods ====================
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "messages": message,
            "model": self.model_name.replace("-FC", ""),
            "temperature": self.temperature,
        }
        if tools and len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        inference_data["tools"] = tools
        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        msg, usage = self._normalize_openai_like_response(api_response)

        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            model_responses = [
                {tc["function"]["name"]: tc["function"].get("arguments", "{}")}
                for tc in tool_calls if tc.get("function")
            ]
            tool_call_ids = [tc.get("id") for tc in tool_calls]
        else:
            model_responses = msg.get("content", "")
            tool_call_ids = []

        reasoning = msg.get("reasoning_content", "")

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": msg,
            "tool_call_ids": tool_call_ids,
            "input_token": (usage or {}).get("prompt_tokens", 0),
            "output_token": (usage or {}).get("completion_tokens", 0),
            "reasoning_content": reasoning,
        }

    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data.setdefault("message", []).extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data.setdefault("message", []).extend(user_message)
        return inference_data

    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data.setdefault("message", []).append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data.get("tool_call_ids", [])
        ):
            inference_data.setdefault("message", []).append(
                {"role": "tool", "content": execution_result, "tool_call_id": tool_call_id}
            )
        return inference_data

    def _add_reasoning_content_if_available_FC(self, api_response: Any, response_data: dict) -> None:
        msg, _ = self._normalize_openai_like_response(api_response)
        if msg.get("tool_calls"):
            assistant_message = {
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": [
                    {
                        "id": tc.get("id"),
                        "type": tc.get("type"),
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in msg["tool_calls"] if tc.get("function")
                ],
            }
            response_data["model_responses_message_for_chat_history"] = assistant_message

    # ==================== Prompting methods ====================
    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"]) }
        return self.generate_with_backoff(
            messages=inference_data["message"],
            model=self.model_name,
            temperature=self.temperature,
        )

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )
        return {"message": []}

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        msg, usage = self._normalize_openai_like_response(api_response)
        reasoning = msg.get("reasoning_content", "")
        return {
            "model_responses": msg.get("content", ""),
            "model_responses_message_for_chat_history": msg,
            "input_token": (usage or {}).get("prompt_tokens", 0),
            "output_token": (usage or {}).get("completion_tokens", 0),
            "reasoning_content": reasoning,
        }

    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data.setdefault("message", []).extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data.setdefault("message", []).extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data.setdefault("message", []).append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data.setdefault("message", []).append(
            {"role": "user", "content": formatted_results_message}
        )
        return inference_data

    def _add_reasoning_content_if_available_prompting(self, api_response: Any, response_data: dict) -> None:
        pass
