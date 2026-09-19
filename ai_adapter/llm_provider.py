from abc import ABC, abstractmethod

# 1. 定义一个统一的响应格式
class UnifiedResponse:
    def __init__(self, content: str,reasoning: str, model: str, usage: dict):
        self.content = content
        self.reasoning = reasoning
        self.model = model  
        self.usage = usage  # 统一为 {'input_tokens': x, 'output_tokens': y}

class LLMProvider(ABC):
    """LLM请求的抽象基类"""
    @abstractmethod
    def generate(self, 
        prompt: str|None, system_pro: str|None, 
        messages: list[dict]|None, 
        **kwargs) -> UnifiedResponse:
        pass

    @abstractmethod
    def window_summary(self, 
        keep_history: int, 
        the_history: list[dict]):
        pass