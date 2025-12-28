import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import hashlib


class SimpleDatabase:
    """简易文件数据库，用于存储数据"""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.employees_file = os.path.join(data_dir, "employees.json")
        self.templates_file = os.path.join(data_dir, "templates.json")
        self.attendance_file = os.path.join(data_dir, "attendance.json")

        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)

        # 初始化文件
        self._init_files()

    def _init_files(self):
        """初始化数据文件"""
        for file_path in [self.employees_file, self.templates_file, self.attendance_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)

    def _read_json(self, file_path: str) -> List:
        """读取JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _write_json(self, file_path: str, data: List):
        """写入JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_employee(self, emp_id: str, name: str, department: str = "", position: str = "") -> bool:
        """添加新员工"""
        employees = self._read_json(self.employees_file)

        # 检查是否已存在
        for emp in employees:
            if emp.get("emp_id") == emp_id:
                return False

        new_employee = {
            "emp_id": emp_id,
            "name": name,
            "department": department,
            "position": position,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active"
        }

        employees.append(new_employee)
        self._write_json(self.employees_file, employees)
        return True

    def add_face_template(self, emp_id: str, embedding_vector: List[float],
                          source: str = "enrollment") -> bool:
        """添加人脸模板（支持多模板存储）"""
        templates = self._read_json(self.templates_file)

        # 生成唯一模板ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        template_hash = hashlib.md5(f"{emp_id}_{timestamp}".encode()).hexdigest()[:8]

        new_template = {
            "template_id": template_hash,
            "emp_id": emp_id,
            "embedding_vector": embedding_vector,
            "source": source,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_active": True
        }

        templates.append(new_template)
        self._write_json(self.templates_file, templates)
        return True

    def get_all_templates(self) -> List[Dict]:
        """获取所有人脸模板"""
        templates = self._read_json(self.templates_file)
        # 只返回激活的模板
        return [t for t in templates if t.get("is_active", True)]

    def get_employee_by_id(self, emp_id: str) -> Optional[Dict]:
        """根据工号获取员工信息"""
        employees = self._read_json(self.employees_file)
        for emp in employees:
            if emp.get("emp_id") == emp_id:
                return emp
        return None

    def add_attendance_record(self, emp_id: str, confidence: float,
                              check_type: str = "in", camera_id: str = "default") -> bool:
        """添加打卡记录"""
        attendance = self._read_json(self.attendance_file)

        new_record = {
            "record_id": len(attendance) + 1,
            "emp_id": emp_id,
            "confidence": confidence,
            "check_type": check_type,
            "camera_id": camera_id,
            "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        attendance.append(new_record)
        self._write_json(self.attendance_file, attendance)
        return True

    def get_employee_templates_count(self, emp_id: str) -> int:
        """获取员工的人脸模板数量"""
        templates = self._read_json(self.templates_file)
        count = 0
        for template in templates:
            if template.get("emp_id") == emp_id and template.get("is_active", True):
                count += 1
        return count

    def delete_employee(self, emp_id: str) -> bool:
        """删除员工（软删除）"""
        employees = self._read_json(self.employees_file)
        updated = False

        for i, emp in enumerate(employees):
            if emp.get("emp_id") == emp_id:
                employees[i]["status"] = "deleted"
                updated = True
                break

        if updated:
            self._write_json(self.employees_file, employees)

        return updated

db = SimpleDatabase()