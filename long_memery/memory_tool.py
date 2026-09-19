from vector_calculation import VectorCalculation

def get_primary_nickname(props: dict) -> str:
    """取出权重最高的那个外号"""
    nicknames = props.get("nicknames", {})
    if not nicknames:
        return None
    return max(nicknames, key=nicknames.get)  # 返回 "小鱼"

def retrieve_start_node(self, query_embedding: list[float], query_norm: float, threshold: float = 0.5) -> str | None:
        best_id = None
        best_sim = -1.0
        for nid, node in self.nodes.items():
            sim = VectorCalculation.cos_calculation(self.norms[nid], query_norm, node.embedding, query_embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = nid
        return best_id if best_sim >= threshold else None

def bfs_activate(self, start_id: str, max_depth: int = 3, fanout: int = 10) -> dict[str, int]:
        """从起点开始 BFS，返回 { 节点ID: 深度 }"""
        if start_id not in self.nodes:
            return {}
        
        activated = {}
        visited = {start_id}
        current_layer = [start_id]
        
        for depth in range(1, max_depth + 1):
            next_layer = []
            for node in current_layer:
                # 获取邻居，按相似度降序排序并剪枝
                neighbors = self.edges.get(node, {})
                sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:fanout]
                
                for neighbor_id, sim in sorted_neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_layer.append(neighbor_id)
                        activated[neighbor_id] = depth
            current_layer = next_layer
            if not current_layer:
                break
        return activated

def update_weights(self, start_id: str, activated: dict[str, int], current_time: float):
        """根据 BFS 结果给节点加权"""
        # 起点额外加 0.5
        if start_id in self.nodes:
            self.nodes[start_id].weight += 0.5
            self.nodes[start_id].last_use_time = current_time
        
        for nid, depth in activated.items():
            # 取边权：如果是直接邻居（depth=1），直接用起点到它的边权
            # 如果是间接邻居，简化处理，取它所有邻居里最高的边权
            if depth == 1:
                sim = self.edges.get(start_id, {}).get(nid, 0.0)
            else:
                # 取该节点所有邻居中最大的边权（代表它最核心的关联）
                neighbor_sims = self.edges.get(nid, {}).values()
                sim = max(neighbor_sims) if neighbor_sims else 0.0
            
            bonus = sim * (0.5 ** depth)
            
            # 1 小时内用过，打折
            if current_time - self.nodes[nid].last_use_time < 3600:
                bonus *= 0.3
            
            self.nodes[nid].weight += bonus
            self.nodes[nid].last_use_time = current_time