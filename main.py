import os
from ai_adapter.deepseek_adapter import DeepSeekAdapter
from ai_adapter.llm_provider import UnifiedResponse 

def main_while(Key):
    SYSTEM_PROMPT = "你是一只活泼可爱的猫娘" # 将来要用于变换人设
    MAX_HISTORY :int = 10 # 最大历史对话轮数，超过就压缩
    KEEP_HISTORY :int = 4 # 保留的最近对话
    history: list[dict] = []
    ds = DeepSeekAdapter(Key)
    while True:
        # 窗口压缩
        if len(history) > MAX_HISTORY:
            ds.window_summary(keep_history=KEEP_HISTORY, the_history=history)
        # 压缩完成
        pr = input("你想对我说什么喵: ")
        if pr == "/STOP":
            return
        current_messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + history + [{"role": "user", "content": pr}]
        a: UnifiedResponse = ds.generate(
            messages=current_messages
        )
        print(f"思考过程：\n{a.reasoning}\n") 
        print(f"输出：\n{a.content}")
        print(f"—— {a.model}\n")
        print(f"input_tokens: {a.usage['input_tokens']}")
        print(f"output: {a.usage['output_tokens']}")
        history.append({"role": "user", "content": pr})      
        history.append({"role": "assistant", "content": a.content})

if __name__ == "__main__":
    KEY = input("APIKEY= ")
    main_while(Key=KEY)
