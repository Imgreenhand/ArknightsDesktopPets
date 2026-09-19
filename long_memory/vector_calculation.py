import numpy as np


class VectorCalculation:
    """纯向量数学，无状态"""

    @staticmethod
    def cos_calculation(norm_a: float, norm_b: float, vec_a: list[float], vec_b: list[float]) -> float:
        """cos<a, b> = a·b / (|a|*|b|)"""
        norm_product: float = norm_a * norm_b
        if norm_product == 0:
            return 0.0  # 防止除以零
        point_product: float = np.dot(vec_a, vec_b)
        return float(point_product / norm_product)

    @staticmethod
    def batch_cos_calculation(matrix: np.ndarray, norms: np.ndarray,
                              query_vec: list[float], query_norm: float) -> np.ndarray:
        """一次性算出 query 与所有节点的相似度

        matrix: (N, D) 的节点向量矩阵
        norms:  (N,)   的节点模长
        返回:   (N,)   的相似度
        """
        denominator = norms * query_norm
        dots = matrix @ np.asarray(query_vec, dtype=float)
        result = np.zeros_like(dots, dtype=float)
        np.divide(dots, denominator, out=result, where=denominator != 0)
        return result
