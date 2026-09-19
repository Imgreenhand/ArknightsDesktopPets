import time
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MemNode:
    text: str                  
    # 作为节点的“标题/摘要”，如人名、事件名
    embedding: list[float]     
    # 文本向量（依然基于 text 生成，保证检索功能）
    node_type: str = "fact"    
    # 新增："person"（人物）, "event"（事件）, "statement"（零散语句）, "fact"（普通事实）
    props: dict = field(default_factory=dict)  
    # 存所有结构化细节
    weight: float = 1.0
    tags: list[str] = field(default_factory=list)
    last_use_time: float = field(default_factory=time.time)

class MemoryGraph:
    """记忆网"""
    def __init__(self):
        self.nodes: dict[str, MemNode] = {}
        # 存放所有节点，例如：
        # {
        #  "mem_1": {
        #  "text": "这是一段记忆",
        #  "embedding": [这是一个存向量的数组],
        #  "weight": 一个浮点数，用来表示记忆的遗忘度，越小记忆越模糊
        #   }
        #  "tags": [这是一个存标签的数组]
        #  "last_use_time": 时间戳
        # }
        self.edges: dict[str, dict[str, float]] = {} # {"mem_1": {"mem_2": 0.5}, ...}
        self.norms: dict[str, float] = {} # 存节点的模长，以免重复计算
        self.num: int = 0 # 先占个位, 表示当前最大节点id, 从JSON中获取
        self.text_to_id: dict[str, str] = {} # 通过文本快速查询索引

    def add_node(self, text: str, embedding: list[float], tags: list[str]|None = None) -> str:
        """添加一个记忆节点，如果文本完全相同则复用已有节点"""
        # 精确文本去重
        if text in self.text_to_id:
            return self.text_to_id[text]  # 直接返回已有的 ID
        
        # 新建节点
        self.num += 1
        nid = f"mem_{self.num}"
        node = MemNode(text=text, embedding=embedding, tags=tags or [])
        self.nodes[nid] = node
        self.norms[nid] = np.linalg.norm(embedding) # 取模长
        self.text_to_id[text] = nid  # 注册索引
        return nid

    def add_memory_pair(self, text_a: str, text_b: str, 
                    emb_a: list[float], emb_b: list[float], 
                    similarity: float) -> tuple[str, str]:
        """添加两个节点并建立双向边，自动去重"""
        # 我c，比我自己想的算法好多了，DS牛逼无需多言
        id_a = self.add_node(text_a, emb_a)
        id_b = self.add_node(text_b, emb_b)
        
        if id_a == id_b:
            return id_a, id_b  # 同一个节点，不加边
        
        # 加双向边
        if id_a not in self.edges:
            self.edges[id_a] = {}
        self.edges[id_a][id_b] = similarity
        
        if id_b not in self.edges:
            self.edges[id_b] = {}
        self.edges[id_b][id_a] = similarity
        
        return id_a, id_b

if __name__ == "__main__":
    graph = MemoryGraph()
    id1 = graph.add_node("螺蛳粉", [1.0, 0.0, 0.0])
    id2 = graph.add_node("柳州", [0.9, 0.1, 0.0])
    graph.add_memory_pair("螺蛳粉", "柳州", [1.0,0,0], [0.9,0.1,0], 0.95)
    activated = graph.bfs_activate(id1, max_depth=2)
    print(activated)  # 应该看到 {'mem_2': 1}