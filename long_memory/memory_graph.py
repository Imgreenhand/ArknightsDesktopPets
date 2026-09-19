"""记忆图：长期记忆的数据结构与激活扩散算法

MemoryGraph 是记忆数据的唯一拥有者，负责：
节点/边的增删、派生索引（文本索引、向量矩阵、模长）、相似度检索、
BFS 激活扩散、权重更新、时间遗忘、JSON 持久化。

自检：python -m long_memory.memory_service
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from .vector_calculation import VectorCalculation


@dataclass
class MemNode:
    text: str
    # 作为节点的“标题/摘要”，如人名、事件名
    embedding: list[float]
    # 文本向量（依然基于 text 生成，保证检索功能）
    node_type: str = "fact"
    # "person"（人物）, "event"（事件）, "statement"（零散语句）, "fact"（普通事实）
    props: dict = field(default_factory=dict)
    # 存所有结构化细节
    tags: list[str] = field(default_factory=list)
    weight: float = 1.0
    # 记忆权重，越大越容易在回忆中被想起
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = 0.0
    # 上次被激活（想起）的时间，0 表示从未激活，仅用于冷却判断
    last_decay_at: float = 0.0
    # 上次结算遗忘的时间，用于避免衰减被重复叠加


class MemoryGraph:
    """记忆网"""

    COLD_DOWN_SECONDS = 3600.0      # 刚被激活过的记忆，在窗口内重复激活要打折
    COLD_DOWN_FACTOR = 0.3
    MAX_WEIGHT = 20.0               # 权重软上限，防止热门记忆无限膨胀
    DECAY_BASE = 0.5                # 每深一层，联想强度乘一次
    DEFAULT_FORGET_TAU = 7 * 24 * 3600.0  # 遗忘时间常数，默认一周

    def __init__(self):
        self.nodes: dict[str, MemNode] = {}
        # { "mem_1": MemNode(...) }
        self.edges: dict[str, dict[str, float]] = {}
        # {"mem_1": {"mem_2": 0.5}, ...} 无权图用相似度当边权
        self.num: int = 0               # 当前最大节点 id 序号，加载存档时恢复
        self._ids: list[str] = []       # 派生索引：节点顺序，与下面的矩阵行一一对应
        self._text_to_id: dict[str, str] = {}
        self._embeddings: np.ndarray | None = None
        self._norms: np.ndarray | None = None

    # ---------- 索引 ----------

    def _reindex(self) -> None:
        """由 nodes 重建全部派生索引，杜绝多份数据不同步"""
        self._ids = list(self.nodes)
        self._text_to_id = {node.text: nid for nid, node in self.nodes.items()}
        if self._ids:
            self._embeddings = np.asarray([self.nodes[nid].embedding for nid in self._ids], dtype=float)
            self._norms = np.linalg.norm(self._embeddings, axis=1)
        else:
            self._embeddings = None
            self._norms = None
        for nid in self._ids:
            if nid.startswith("mem_"):
                try:
                    self.num = max(self.num, int(nid[4:]))
                except ValueError:
                    pass

    def _append_index(self, nid: str, node: MemNode) -> None:
        """新增节点时就地追加索引，避免每次都全量重建"""
        vector = np.asarray(node.embedding, dtype=float)
        if self._embeddings is not None and vector.shape[0] != self._embeddings.shape[1]:
            raise ValueError(f"向量维度不一致：已有 {self._embeddings.shape[1]} 维，收到 {vector.shape[0]} 维")
        self._ids.append(nid)
        self._text_to_id[node.text] = nid
        self._embeddings = vector[None, :] if self._embeddings is None else np.vstack([self._embeddings, vector])
        norm = float(np.linalg.norm(vector))
        self._norms = np.array([norm]) if self._norms is None else np.append(self._norms, norm)

    # ---------- 增删 ----------

    def add_node(self, text: str, embedding: list[float], tags: list[str] | None = None,
                 node_type: str = "fact", props: dict | None = None) -> str:
        """添加一个记忆节点，文本完全相同则复用已有节点（并补齐标签与属性）"""
        if text in self._text_to_id:
            nid = self._text_to_id[text]
            node = self.nodes[nid]
            if tags:
                node.tags = list(dict.fromkeys(node.tags + list(tags)))
            if props:
                node.props.update(props)
            return nid

        self.num += 1
        nid = f"mem_{self.num}"
        self.nodes[nid] = MemNode(text=text, embedding=embedding, node_type=node_type,
                                  props=props or {}, tags=tags or [])
        self._append_index(nid, self.nodes[nid])
        return nid

    def add_edge(self, id_a: str, id_b: str, similarity: float) -> None:
        """建立双向边；同一个节点或已有更高边权时不重复覆盖"""
        if id_a == id_b or id_a not in self.nodes or id_b not in self.nodes:
            return
        self.edges.setdefault(id_a, {})[id_b] = similarity
        self.edges.setdefault(id_b, {})[id_a] = similarity

    def add_memory_pair(self, text_a: str, text_b: str,
                        emb_a: list[float], emb_b: list[float],
                        similarity: float) -> tuple[str, str]:
        """添加两个节点并建立双向边，自动去重"""
        id_a = self.add_node(text_a, emb_a)
        id_b = self.add_node(text_b, emb_b)
        self.add_edge(id_a, id_b, similarity)
        return id_a, id_b

    def remove_nodes(self, node_ids: list[str]) -> None:
        """删除节点及其所有连边，并重建索引"""
        for nid in list(node_ids):
            self.nodes.pop(nid, None)
            self.edges.pop(nid, None)
        for neighbors in self.edges.values():
            for nid in node_ids:
                neighbors.pop(nid, None)
        self._reindex()

    # ---------- 检索与扩散 ----------

    def retrieve(self, query_embedding: list[float], threshold: float = 0.5,
                 top_k: int = 3) -> list[tuple[str, float]]:
        """检索与 query 最相似的记忆：(节点ID, 相似度)，按相似度降序"""
        if self._embeddings is None:
            return []
        query_vec = np.asarray(query_embedding, dtype=float)
        if query_vec.shape[0] != self._embeddings.shape[1]:
            raise ValueError(f"查询向量维度不一致：已有 {self._embeddings.shape[1]} 维，收到 {query_vec.shape[0]} 维")
        query_norm = float(np.linalg.norm(query_vec))
        sims = VectorCalculation.batch_cos_calculation(self._embeddings, self._norms, query_vec, query_norm)

        # 用稳定排序：相似度并列时优先返回更早记住的那条，保证结果可复现
        order = np.argsort(-sims, kind="stable")[:top_k]
        return [(self._ids[int(i)], float(sims[int(i)])) for i in order if sims[int(i)] >= threshold]

    def bfs_activate(self, start_id: str, max_depth: int = 3,
                     fanout: int = 10) -> dict[str, tuple[int, float]]:
        """从起点开始激活扩散，返回 {节点ID: (深度, 入边相似度)}"""
        if start_id not in self.nodes:
            return {}

        activated: dict[str, tuple[int, float]] = {}
        visited = {start_id}
        current_layer = [start_id]

        for depth in range(1, max_depth + 1):
            next_layer = []
            for node_id in current_layer:
                # 邻居按边权降序，只取尚未访问的前 fanout 个，避免名额被已访问节点占掉
                ranked = sorted(self.edges.get(node_id, {}).items(), key=lambda kv: kv[1], reverse=True)
                picked = 0
                for neighbor_id, similarity in ranked:
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    next_layer.append(neighbor_id)
                    activated[neighbor_id] = (depth, similarity)
                    picked += 1
                    if picked >= fanout:
                        break
            current_layer = next_layer
            if not current_layer:
                break
        return activated

    def update_weights(self, start_id: str, activated: dict[str, tuple[int, float]],
                       current_time: float | None = None) -> None:
        """按扩散结果加权：起点额外 +0.5（触景生情），其余按 入边相似度 * 0.5^深度 衰减"""
        now = time.time() if current_time is None else current_time

        if start_id in self.nodes:
            self.nodes[start_id].weight = min(self.nodes[start_id].weight + 0.5, self.MAX_WEIGHT)
            self.nodes[start_id].last_activated_at = now

        for nid, (depth, edge_similarity) in activated.items():
            node = self.nodes.get(nid)
            if node is None:
                continue
            bonus = edge_similarity * (self.DECAY_BASE ** depth)
            # 刚用过又想起来：说明在重复同一个话题，奖励打折
            if now - node.last_activated_at < self.COLD_DOWN_SECONDS:
                bonus *= self.COLD_DOWN_FACTOR
            node.weight = min(node.weight + bonus, self.MAX_WEIGHT)
            node.last_activated_at = now

    # ---------- 遗忘 ----------

    def apply_forgetting(self, current_time: float | None = None,
                         tau: float | None = None,
                         prune_floor: float | None = None) -> list[str]:
        """让记忆随时间淡出：weight *= exp(-Δt / tau)

        Δt 从上一次“被激活或上次结算遗忘”算起，因此可重复调用而不会重复叠加衰减。
        prune_floor 不为 None 时，权重低于该阈值的记忆会被彻底清理，返回被清理的节点 ID。
        """
        now = time.time() if current_time is None else current_time
        if tau is None:
            tau = self.DEFAULT_FORGET_TAU

        faint: list[str] = []
        for nid, node in self.nodes.items():
            baseline = max(node.created_at, node.last_activated_at, node.last_decay_at)
            node.weight *= math.exp(-(now - baseline) / tau)
            node.last_decay_at = now
            if prune_floor is not None and node.weight < prune_floor:
                faint.append(nid)

        if faint:
            self.remove_nodes(faint)
        return faint

    # ---------- 持久化 ----------

    def to_dict(self) -> dict:
        known = {f.name for f in fields(MemNode)}
        return {
            "version": 1,
            "num": self.num,
            "nodes": {nid: {k: v for k, v in node.__dict__.items() if k in known}
                      for nid, node in self.nodes.items()},
            "edges": {nid: dict(neighbors) for nid, neighbors in self.edges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryGraph":
        """只恢复 nodes/edges，派生索引全部重建，num 取存档与现存 id 的较大值"""
        graph = cls()
        known = {f.name for f in fields(MemNode)}
        for nid, payload in data.get("nodes", {}).items():
            graph.nodes[nid] = MemNode(**{k: v for k, v in payload.items() if k in known})
        for nid, neighbors in data.get("edges", {}).items():
            graph.edges[nid] = {k: float(v) for k, v in neighbors.items()}
        graph.num = int(data.get("num", 0))
        graph._reindex()
        return graph

    def save(self, path: str | Path) -> None:
        """落盘：先写临时文件再替换，避免中途崩溃写坏存档"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @classmethod
    def load(cls, path: str | Path) -> "MemoryGraph":
        """读档；文件不存在时返回一张空图"""
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
