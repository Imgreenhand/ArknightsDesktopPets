import numpy as np

class VectorCalculation:

    @staticmethod
    def cos_calculation(norm_a: float, norm_b: float, vec_a: list[float], vec_b: list[float]) -> float:
        """cos<a, b> = a·b/|a|*|b|"""
        norm_product: float = norm_a * norm_b
        if norm_product == 0:
            return 0.0  # 防止除以零
        point_product: float = np.dot(vec_a, vec_b)
        return point_product/norm_product