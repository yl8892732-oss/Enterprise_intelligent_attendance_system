import numpy as np
from typing import List, Tuple, Dict, Optional
from database import db


class FaceRecognizer:
    """人脸识别器 - 核心比对算法"""

    def __init__(self, threshold: float = 0.7):
        """
        初始化人脸识别器

        Args:
            threshold: 相似度阈值，默认0.7
        """
        self.threshold = threshold
        self.templates = []  # 存储所有人脸模板
        self._load_templates()

        # 外部模块的函数引用（等待陈锡翘和黄晨禹提供）
        self.preprocess_func = None
        self.extract_feature_func = None

    def _load_templates(self):
        """从数据库加载所有人脸模板"""
        self.templates = db.get_all_templates()
        print(f"✅ 已加载 {len(self.templates)} 个人脸模板")

    def refresh_templates(self):
        """刷新模板数据（当有新员工注册时调用）"""
        self._load_templates()
        print("🔄 人脸模板已刷新")

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            float: 相似度值，范围[-1, 1]
        """
        # 转换为numpy数组
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        # 计算点积
        dot_product = np.dot(v1, v2)

        # 计算模长
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # 避免除零错误
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)

        # 确保在[-1, 1]范围内
        return max(-1.0, min(1.0, similarity))

    def identify(self, query_vector: List[float]) -> Tuple[str, float, bool, Dict]:
        """
        识别单个人脸

        Args:
            query_vector: 查询向量（512维）

        Returns:
            tuple: (员工ID, 相似度, 是否识别成功, 员工信息)
        """
        if not self.templates:
            return "Unknown", 0.0, False, {}

        best_match = None
        best_similarity = -1.0

        # 遍历所有模板，找到最相似的
        for template in self.templates:
            template_vector = template.get("embedding_vector", [])

            if len(template_vector) != len(query_vector):
                continue

            # 调用你的比对算法
            similarity = self.cosine_similarity(query_vector, template_vector)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = template

        # 判断是否达到阈值
        recognized = best_similarity >= self.threshold

        if recognized and best_match:
            emp_id = best_match.get("emp_id", "Unknown")
            employee_info = db.get_employee_by_id(emp_id) or {}
            return emp_id, best_similarity, True, employee_info
        else:
            return "Unknown", best_similarity, False, {}

    def search_top_k(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        """
        搜索最相似的K个人脸

        Args:
            query_vector: 查询向量
            k: 返回的结果数量

        Returns:
            list: 相似度最高的K个结果
        """
        if not self.templates:
            return []

        results = []

        for template in self.templates:
            template_vector = template.get("embedding_vector", [])

            if len(template_vector) != len(query_vector):
                continue

            similarity = self.cosine_similarity(query_vector, template_vector)

            results.append({
                "emp_id": template.get("emp_id"),
                "similarity": similarity,
                "template_id": template.get("template_id")
            })

        # 按相似度降序排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # 返回前K个结果
        return results[:k]

    def set_preprocess_function(self, func):
        """设置陈锡翘的预处理函数"""
        self.preprocess_func = func
        print("✅ 已设置陈锡翘的预处理函数")

    def set_extract_feature_function(self, func):
        """设置黄晨禹的特征提取函数"""
        self.extract_feature_func = func
        print("✅ 已设置黄晨禹的特征提取函数")

    def process_image(self, image_data) -> Tuple[str, float, bool, Dict]:
        """
        处理单张图片（完整流程）

        Args:
            image_data: 图片数据（Base64字符串或numpy数组）

        Returns:
            tuple: 识别结果
        """
        if self.preprocess_func is None:
            raise ValueError("请先设置陈锡翘的预处理函数")
        if self.extract_feature_func is None:
            raise ValueError("请先设置黄晨禹的特征提取函数")

        # 1. 调用陈锡翘的预处理
        processed_faces = self.preprocess_func(image_data)

        if not processed_faces:
            return "Unknown", 0.0, False, {}

        # 2. 调用黄晨禹的特征提取
        face_vectors = []
        for face in processed_faces:
            vector = self.extract_feature_func(face)
            face_vectors.append(vector)

        # 3. 调用你的比对算法（取第一个脸）
        if face_vectors:
            return self.identify(face_vectors[0])

        return "Unknown", 0.0, False, {}


recognizer = FaceRecognizer(threshold=0.7)