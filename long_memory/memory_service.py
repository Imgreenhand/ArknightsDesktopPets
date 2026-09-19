"""记忆服务门面：把“文本 → 向量 → 检索 → 扩散 → 加权 → 注入提示词”串起来

自检：python -m long_memory.memory_service
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from .memory_graph import MemoryGraph


class Embedder(Protocol):
    """向量化接口

    DeepSeek 目前没有 embedding 接口，所以具体实现由调用方注入，
    真实使用时换成任意向量服务即可，本模块不绑定具体供应商。
    """

    def embed(self, text: str) -> list[float]:
        ...


def get_primary_nickname(props: dict) -> str | None:
    """取出权重最高的那个外号"""
    nicknames = props.get("nicknames", {})
    if not nicknames:
        return None
    return max(nicknames, key=nicknames.get)


class MemoryService:
    """桌宠的长期记忆服务"""

    def __init__(self, embedder: Embedder, store_path: str | Path | None = None,
                 start_threshold: float = 0.5, link_threshold: float = 0.5,
                 forget_tau: float | None = None, forget_prune_floor: float | None = 0.05):
        self.embedder = embedder
        self.store_path = Path(store_path) if store_path else None
        self.graph = MemoryGraph.load(self.store_path) if self.store_path else MemoryGraph()
        self.start_threshold = start_threshold   # 检索起点所需的最低相似度
        self.link_threshold = link_threshold     # 两条记忆之间要有多像才连边
        self.forget_tau = forget_tau             # None 表示用 MemoryGraph 的默认值
        self.forget_prune_floor = forget_prune_floor

    # ---------- 写入 ----------

    def remember(self, text: str, node_type: str = "fact",
                 tags: list[str] | None = None, props: dict | None = None,
                 link_top_k: int = 5) -> str:
        """记下一句话：向量化 → 建点 → 与已有记忆按相似度连边 → 落盘"""
        embedding = self.embedder.embed(text)
        nid = self.graph.add_node(text, embedding, tags=tags, node_type=node_type, props=props)

        # 与已有记忆建立相似度边（自己除外）
        for other_id, similarity in self.graph.retrieve(embedding, threshold=self.link_threshold,
                                                        top_k=link_top_k + 1):
            if other_id != nid:
                self.graph.add_edge(nid, other_id, similarity)

        self._save()
        return nid

    # ---------- 回忆 ----------

    def recall(self, query: str, top_k: int = 3, max_depth: int = 3,
               result_limit: int = 5) -> list[dict]:
        """回忆：先结算遗忘 → 检索起点 → 激活扩散 → 加权 → 返回权重最高的若干条记忆"""
        self.graph.apply_forgetting(tau=self.forget_tau, prune_floor=self.forget_prune_floor)

        starts = self.graph.retrieve(self.embedder.embed(query), threshold=self.start_threshold, top_k=top_k)
        if not starts:
            return []

        now = time.time()
        touched: dict[str, None] = {}
        for start_id, _ in starts:
            touched[start_id] = None
            activated = self.graph.bfs_activate(start_id, max_depth=max_depth)
            self.graph.update_weights(start_id, activated, current_time=now)
            for nid in activated:
                touched[nid] = None

        self._save()

        memories = [
            {
                "id": nid,
                "text": node.text,
                "weight": round(node.weight, 4),
                "node_type": node.node_type,
                "tags": node.tags,
                "depth": 0,
            }
            for nid in touched
            if (node := self.graph.nodes.get(nid)) is not None
        ]
        memories.sort(key=lambda m: m["weight"], reverse=True)
        return memories[:result_limit]

    def format_for_prompt(self, memories: list[dict], header: str = "【你脑海中浮现的记忆】") -> str:
        """把回忆结果拼成可直接注入提示词的片段"""
        if not memories:
            return ""
        lines = [f"- {m['text']}" for m in memories]
        return header + "\n" + "\n".join(lines)

    def _save(self) -> None:
        if self.store_path:
            self.graph.save(self.store_path)


if __name__ == "__main__":
    # 自检：建点、去重、连边、扩散、加权、存盘再加载
    import tempfile


    class _DemoEmbedder:
        """演示用的假向量器：按关键词生成 3 维向量，真实使用请注入真向量服务"""

        TABLE = {"螺蛳粉": 0, "柳州": 0, "猫": 1, "狗": 1, "python": 2, "代码": 2}

        def embed(self, text: str) -> list[float]:
            vector = [0.0, 0.0, 0.0]
            for keyword, index in self.TABLE.items():
                if keyword in text:
                    vector[index] += 1.0
            return vector if any(vector) else [0.1, 0.1, 0.1]


    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "memory.json"
        service = MemoryService(_DemoEmbedder(), store_path=store)

        service.remember("我最喜欢吃螺蛳粉，尤其是柳州的")
        service.remember("柳州是个好地方")
        service.remember("我喜欢写 python 代码")
        assert len(service.graph.nodes) == 3, "三段不同文本应该建三个节点"

        service.remember("柳州是个好地方")  # 完全相同的文本
        assert len(service.graph.nodes) == 3, "相同文本不应重复建点"

        hits = service.recall("螺蛳粉", top_k=1)
        assert hits and "螺蛳粉" in hits[0]["text"], f"最相关的记忆没排第一：{hits}"
        assert len(hits) == 2, f"扩散应把「柳州」一起带出来：{hits}"

        reloaded = MemoryGraph.load(store)
        assert len(reloaded.nodes) == len(service.graph.nodes), "存盘再加载，节点数应一致"
        assert set(reloaded.edges) == set(service.graph.edges), "存盘再加载，边应完整恢复"
        assert reloaded.retrieve([1.0, 0.0, 0.0], threshold=0.5, top_k=1), "重建索引后应能正常检索"

        print("自检通过：", [m["text"] for m in hits])
