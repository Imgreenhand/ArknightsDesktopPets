import os
from openai import OpenAI, OpenAIError
from .llm_provider import LLMProvider, UnifiedResponse

class DeepSeekAdapter(LLMProvider):
    """DeepSeek请求"""
    def __init__(self, key: str, url: str="https://api.deepseek.com"):
        self.client = OpenAI(api_key=key, base_url=url)

    def generate(self, prompt=None, system_pro=None, messages=None, **kwargs):
        # 判断调用方式是单句还是对话历史
        if messages is not None:
            (_messages := messages)
        else:
            if prompt is None:
                return UnifiedResponse(
                    content=f"调用处缺少关键参数prompt，无法构造请求喵",
                    reasoning="", 
                    model="error",
                    usage={
                        'input_tokens': 0, 
                        'output_tokens': 0
                    }
                )
            else:
                (_messages := [
                    {"role": "system", "content": system_pro},
                    {"role": "user", "content": prompt},
                ])
        # 尝试发送请求
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get('model', 'deepseek-chat'),
                messages=_messages,
                stream=False,
                reasoning_effort=kwargs.get('reasoning_effort', 'high'),
                extra_body={"thinking": {"type": "enabled"}}
            )
            return UnifiedResponse(
                content=response.choices[0].message.content,
                reasoning=response.choices[0].message.reasoning_content, 
                model=response.model,
                usage={
                    'input_tokens': response.usage.prompt_tokens, 
                    'output_tokens': response.usage.completion_tokens
                }
            )
        # 如果失败返回错误码
        except OpenAIError as e:
            # 尝试从 OpenAI 库抛出的异常里提取后端给的真实错误文案
            error_msg = str(e)
            # 如果异常对象有 response 属性（OpenAI 的 APIError 通常会有）
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_json = e.response.json()
                    error_msg = err_json.get('error', {}).get('message', error_msg)
                except:
                    pass
            # 再针对常见错误码（如 401）给用户更明确的指引
            if "401" in str(e) or "Authentication" in str(e):
                friendly_content = f"呜……API Key 认证失败喵（{error_msg}）。请检查你的密钥是不是复制全了，或者重新生成一个试试～"
            else:
                friendly_content = f"网络连接出了点问题喵：{error_msg}。稍等片刻再试试？"
            return UnifiedResponse(
                content=friendly_content,
                reasoning="", 
                model="error",
                usage={
                    'input_tokens': 0, 
                    'output_tokens': 0
                }
            )

    def window_summary(self, keep_history, the_history):
        """窗口压缩"""
        KEEP_HISTORY = keep_history
        history = the_history
        print("对话长度达到限制，正在压缩中喵...")
        keep_messages = history[-KEEP_HISTORY:]
        to_summarize = history[:-KEEP_HISTORY]
        summary_messages = [
            {"role": "system", 
             "content": 
             "你是一个精准的对话总结助手, 请用一段话概括用户的对话内容, 学习进度和已讨论的核心主题。"}
        ] + to_summarize + [
            {"role": "user", 
             "content": 
             "请总结以上对话的核心内容, 不要遗漏重要知识点, 保留关键信息, 去掉废话、重复内容：。"}
        ]
        summary_response = self.generate(messages=summary_messages)
        summary_content = summary_response.content
        new_history = [
            {"role": "user", "content": "以下是对之前对话的总结："},
            {"role": "assistant", "content": summary_content}
        ] + keep_messages
        the_history[:] = new_history
        print(f"总结完成，内容如下喵：\n{summary_content}\n")