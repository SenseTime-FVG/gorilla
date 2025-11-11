import json
import os
import time
from numpy import extract
import requests
from typing import Any, Optional, Tuple, List
import re

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    func_doc_language_specific_pre_processing,
)

class LightLLMHandler(OSSHandler):
    """
    LightLLM本地推理处理器
    
    这个处理器用于测试通过LightLLM部署的本地模型。
    支持两种模式：
    - 提示模式（Prompt Mode）：传统的提示-响应模式
    - 函数调用模式（FC Mode）：支持工具/函数调用的模式
    
    使用方法：
    1. 启动LightLLM服务
    2. 通过命令行参数或环境变量设置服务地址
    3. 使用 --skip-server-setup 参数跳过服务器设置
    """

    def __init__(self, model_name, temperature, args) -> None:
        super().__init__(model_name, temperature)
        
        self.model_name = model_name
        
        # 从环境变量获取LightLLM服务配置（支持命令行参数覆盖）
        self.lightllm_url = args.lightllm_url
        
        # 设置生成参数（支持外部传入）
        self.temperature = temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        self.stop_tokens = args.stop_tokens
        self.do_sample = args.do_sample
        self.skip_special_tokens = args.skip_special_tokens
        self.add_special_tokens = args.add_special_tokens
        self.enable_thinking = args.enable_thinking
    
    @override
    def decode_ast(self, result, language="Python", has_tool_call_tag=True):
        # Model response is of the form:
        # "<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Taylor Swift\", \"duration\": 20}}\n</tool_call>\n<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Maroon 5\", \"duration\": 15}}\n</tool_call>"?
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            return []
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]


    @override
    def decode_execute(self, result, has_tool_call_tag=True):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            return []
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)


    @override
    def _format_prompt(self, messages, function):
        """
        "chat_template":
        {%- if tools %}
            {{- '<|im_start|>system\n' }}
            {%- if messages[0].role == 'system' %}
                {{- messages[0].content + '\n\n' }}
            {%- endif %}
            {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
            {%- for tool in tools %}
                {{- "\n" }}
                {{- tool | tojson }}
            {%- endfor %}
            {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
        {%- else %}
            {%- if messages[0].role == 'system' %}
                {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
        {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
        {%- for message in messages[::-1] %}
            {%- set index = (messages|length - 1) - loop.index0 %}
            {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
                {%- set ns.multi_step_tool = false %}
                {%- set ns.last_query_index = index %}
            {%- endif %}
        {%- endfor %}
        {%- for message in messages %}
            {%- if message.content is string %}
                {%- set content = message.content %}
            {%- else %}
                {%- set content = '' %}
            {%- endif %}
            {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
                {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
            {%- elif message.role == "assistant" %}
                {%- set reasoning_content = '' %}
                {%- if message.reasoning_content is string %}
                    {%- set reasoning_content = message.reasoning_content %}
                {%- else %}
                    {%- if '</think>' in content %}
                        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                    {%- endif %}
                {%- endif %}
                {%- if loop.index0 > ns.last_query_index %}
                    {%- if loop.last or (not loop.last and reasoning_content) %}
                        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
                    {%- else %}
                        {{- '<|im_start|>' + message.role + '\n' + content }}
                    {%- endif %}
                {%- else %}
                    {{- '<|im_start|>' + message.role + '\n' + content }}
                {%- endif %}
                {%- if message.tool_calls %}
                    {%- for tool_call in message.tool_calls %}
                        {%- if (loop.first and content) or (not loop.first) %}
                            {{- '\n' }}
                        {%- endif %}
                        {%- if tool_call.function %}
                            {%- set tool_call = tool_call.function %}
                        {%- endif %}
                        {{- '<tool_call>\n{"name": "' }}
                        {{- tool_call.name }}
                        {{- '", "arguments": ' }}
                        {%- if tool_call.arguments is string %}
                            {{- tool_call.arguments }}
                        {%- else %}
                            {{- tool_call.arguments | tojson }}
                        {%- endif %}
                        {{- '}\n</tool_call>' }}
                    {%- endfor %}
                {%- endif %}
                {{- '<|im_end|>\n' }}
            {%- elif message.role == "tool" %}
                {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
                    {{- '<|im_start|>user' }}
                {%- endif %}
                {{- '\n<tool_response>\n' }}
                {{- content }}
                {{- '\n</tool_response>' }}
                {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
                    {{- '<|im_end|>\n' }}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
            {%- endif %}
        {%- endif %}
        """
        formatted_prompt = ""
       
        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"].strip() + "\n\n"
            if self.enable_thinking:
                formatted_prompt += "Reason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end.\n\n"
            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'

        else:
            if messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )
            if self.enable_thinking:
                formatted_prompt += 'Reason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end.'

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]

                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + f"\n</think>\n\n"
                            + content.lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"
                
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"
                        
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        
                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '
                        
                        if isinstance(tool_call["arguments"], str):
                            formatted_prompt += tool_call["arguments"]
                        else:
                            formatted_prompt += json.dumps(tool_call["arguments"])
                        
                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    @override
    def _query_prompting(self, inference_data: dict):
        """
        重写查询方法，使用LightLLM的generate接口
        """
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        # 准备LightLLM generate接口的参数
        generate_params = {
            "inputs": formatted_prompt,
            "parameters":{
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "max_new_tokens": self.max_new_tokens,
                "stop_sequences": [self.stop_tokens],
                "do_sample": self.do_sample,
                "skip_special_tokens": self.skip_special_tokens,
                "add_special_tokens": self.add_special_tokens,
            }
            
        }

        # 调用LightLLM generate接口
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.lightllm_url}",
                data=json.dumps(generate_params),
                # timeout=72000,  # 避免超时错误
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            api_response = response.json()
        except requests.exceptions.RequestException as e:
            # 1. 统一异常捕获
            import traceback
            tb = traceback.format_exc()
            # 2. 返回的响应信息中的日志
            e_response = getattr(e, "response", None)
            response_text = getattr(e_response, "text", "N/A")
            if response_text and len(response_text) > 800:
                response_text = response_text[:800] + "...[TRUNCATED]"
            # 3. 一般错误都是websearch的时候在fetchurl时返回了过长的数据，基本都是message最后一条
            # 所以这边返回一下message的倒数第二条看看fetch了哪个网址
            last_assistant = ""
            try:
                if message and len(message) > 1:
                    last_assistant = json.dumps(message[-2])
            except Exception as ee:
                last_assistant = f"some error when getting last assistant: {ee}"
            # 组织所有异常信息并返回
            err_message = {
                "tb": tb,
                "response_text": response_text,
                "last_assistant": last_assistant,
            }
            raise Exception(f"LightLLM generate接口调用失败: {json.dumps(err_message)}")
        
        end_time = time.time()
        query_latency = end_time - start_time

        print(api_response)
        return api_response, query_latency


    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:  
        model_response = api_response['generated_text'][0]
        extracted_tool_calls = self._extract_tool_calls(model_response)
        # extract_content, extracted_tool_calls = self._extract_content_and_tool_calls(model_response)

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "reasoning_content": reasoning_content,
                "content": "",
                "tool_calls": extracted_tool_calls,
            }
            # model_responses_message_for_chat_history = {
            #     "role": "assistant",
            #     "content": extract_content,
            #     "tool_calls": extracted_tool_calls,
            # }

        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "reasoning_content": reasoning_content,
                "content": cleaned_response,
            }

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response['prompt_tokens'],
            "output_token": api_response['count_output_tokens'],
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )

        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Process matches into a list of dictionaries
        result = []
        for match in matches:
            try:
                match = json.loads(match.strip())
            except Exception as e:
                pass
            result.append(match)
        return result

    @staticmethod
    def _extract_content_and_tool_calls(input_string: str) -> Tuple[str, List[dict]]:
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)
        # 去除所有 tool_call 标签块，得到纯文本 content
        content_text = re.sub(pattern, "", input_string, flags=re.DOTALL).strip()

        tool_calls: List[dict] = []
        for raw in matches:
            raw = raw.strip()
            try:
                parsed = json.loads(raw.strip())
            except Exception:
                # JSON 解析失败时，按原始字符串放入 arguments
               pass

            tool_calls.append(parsed)

        return content_text, tool_calls
        
    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        
        # test_entry["question"][0] = system_prompt_pre_processing_lightllm_model(
        #     test_entry["question"][0], functions, test_category
        # )

        return {"message": [], "function": functions}    
    