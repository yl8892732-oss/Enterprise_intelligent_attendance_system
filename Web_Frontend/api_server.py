from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import traceback

# 导入自定义模块
from database import db
from face_recognizer import recognizer

# 创建Flask应用
app = Flask(__name__)

# 解决跨域问题（前端可以独立运行在本地）
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:*", "http://127.0.0.1:*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


@app.route('/')
def index():
    """首页"""
    return """
    <h1>人脸考勤系统API</h1>
    <p>开发者：盘先琰</p>
    <p>API接口：</p>
    <ul>
        <li>POST /api/enroll - 员工注册</li>
        <li>POST /api/identify - 人脸识别</li>
        <li>GET /api/health - 健康检查</li>
        <li>GET /api/test - 连通测试</li>
    </ul>
    """


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "success",
        "message": "API服务运行正常",
        "version": "1.0.0",
        "developer": "盘先琰"
    })


@app.route('/api/test', methods=['GET'])
def test_connection():
    """测试接口 - 用于前端确认连通性"""
    return jsonify({
        "status": "success",
        "message": "前端连接测试成功！",
        "api_url": "http://127.0.0.1:5000",
        "endpoints": [
            {"method": "POST", "path": "/api/enroll", "desc": "员工注册"},
            {"method": "POST", "path": "/api/identify", "desc": "人脸识别"},
            {"method": "GET", "path": "/api/health", "desc": "健康检查"}
        ]
    })


@app.route('/api/enroll', methods=['POST'])
def enroll_employee():
    """
    员工注册接口
    严格按照前端要求的格式实现！

    前端发送的JSON格式：
    {
        "name": "员工姓名",
        "emp_id": "员工工号",
        "image": "Base64编码的图片数据"
    }

    返回格式：
    成功(200): {"status": "success", "message": "注册成功"}
    失败(400/500): {"status": "error", "message": "失败原因"}
    """
    try:
        # 获取前端发送的数据
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "未接收到数据"
            }), 400

        # 检查必要字段
        required_fields = ['name', 'emp_id', 'image']
        missing_fields = []

        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"缺少必要字段: {', '.join(missing_fields)}"
            }), 400

        # 提取数据
        name = data['name'].strip()
        emp_id = data['emp_id'].strip()
        image_base64 = data['image']

        print(f"收到注册请求: {name} ({emp_id})")

        # 1. 解码Base64图片
        try:
            # 移除可能的Base64前缀
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]

            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))

            # 转换为RGB（确保三通道）
            if image.mode != 'RGB':
                image = image.convert('RGB')

        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"图片解码失败: {str(e)}"
            }), 400

        # 2. 调用陈锡翘的预处理函数（等他提供）
        # processed_faces = recognizer.preprocess_func(image)
        # 暂时跳过，用随机向量代替

        # 3. 调用黄晨禹的特征提取函数（等他提供）
        # face_vector = recognizer.extract_feature_func(processed_faces[0])
        # 暂时用随机向量代替
        import random
        face_vector = [random.uniform(-1, 1) for _ in range(512)]

        # 4. 保存到数据库
        # 4.1 保存员工信息
        success = db.add_employee(emp_id, name)

        if not success:
            return jsonify({
                "status": "error",
                "message": f"员工工号 {emp_id} 已存在"
            }), 400

        # 4.2 保存人脸模板
        db.add_face_template(emp_id, face_vector, "enrollment")

        # 5. 刷新识别器的模板缓存
        recognizer.refresh_templates()

        print(f"✅ 员工注册成功: {name} ({emp_id})")

        return jsonify({
            "status": "success",
            "message": f"员工 {name} 注册成功"
        }), 200

    except Exception as e:
        # 记录详细错误信息
        error_trace = traceback.format_exc()
        print(f"注册接口错误: {str(e)}")
        print(f"错误详情:\n{error_trace}")

        return jsonify({
            "status": "error",
            "message": f"注册失败: {str(e)}"
        }), 500


@app.route('/api/identify', methods=['POST'])
def identify_face():
    """
    人脸识别接口
    用于实时打卡识别

    请求格式：
    {
        "face_vector": [0.1, 0.2, ...]  # 512维特征向量
    }

    返回格式：
    {
        "emp_id": "员工工号",
        "confidence": 0.85,
        "recognized": true/false
    }
    """
    try:
        data = request.get_json()

        if not data or 'face_vector' not in data:
            return jsonify({
                "emp_id": "Unknown",
                "confidence": 0.0,
                "recognized": False,
                "error": "缺少人脸向量数据"
            }), 200  # 返回200，但recognized为False

        face_vector = data['face_vector']
        threshold = data.get('threshold', 0.7)

        # 设置临时阈值
        original_threshold = recognizer.threshold
        recognizer.threshold = threshold

        # 调用识别函数
        emp_id, confidence, recognized, employee_info = recognizer.identify(face_vector)

        # 恢复原始阈值
        recognizer.threshold = original_threshold

        response = {
            "emp_id": emp_id,
            "confidence": float(confidence),
            "recognized": recognized
        }

        if recognized and employee_info:
            response["employee_info"] = {
                "name": employee_info.get("name", ""),
                "department": employee_info.get("department", ""),
                "position": employee_info.get("position", "")
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "emp_id": "Unknown",
            "confidence": 0.0,
            "recognized": False,
            "error": str(e)
        }), 200


@app.route('/api/attendance', methods=['POST'])
def record_attendance():
    """
    打卡记录接口
    由汤艾梧在连续K帧识别成功后调用

    请求格式：
    {
        "emp_id": "员工工号",
        "confidence": 0.85,
        "check_type": "in"  # 上班: "in", 下班: "out"
    }
    """
    try:
        data = request.get_json()

        required_fields = ['emp_id', 'confidence']
        missing_fields = [f for f in required_fields if f not in data]

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"缺少必要字段: {', '.join(missing_fields)}"
            }), 400

        emp_id = data['emp_id']
        confidence = data['confidence']
        check_type = data.get('check_type', 'in')

        # 检查员工是否存在
        employee = db.get_employee_by_id(emp_id)
        if not employee:
            return jsonify({
                "status": "error",
                "message": f"员工 {emp_id} 不存在"
            }), 404

        # 记录打卡
        db.add_attendance_record(emp_id, confidence, check_type)

        return jsonify({
            "status": "success",
            "message": f"打卡成功: {employee.get('name', emp_id)}",
            "check_type": check_type,
            "time": db._read_json(db.attendance_file)[-1]["check_time"]
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"打卡失败: {str(e)}"
        }), 500


@app.route('/api/employees', methods=['GET'])
def get_employees():
    """获取所有员工列表"""
    try:
        employees = db._read_json(db.employees_file)
        # 只返回活跃员工
        active_employees = [e for e in employees if e.get("status") != "deleted"]

        # 统计模板数量
        for emp in active_employees:
            emp_id = emp.get("emp_id")
            emp["template_count"] = db.get_employee_templates_count(emp_id)

        return jsonify({
            "status": "success",
            "count": len(active_employees),
            "employees": active_employees
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"获取员工列表失败: {str(e)}"
        }), 500


@app.after_request
def after_request(response):
    """添加CORS头到所有响应"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


if __name__ == '__main__':
    print("=" * 60)
    print("人脸考勤系统API服务器")
    print("开发者：帅哥美女们")
    print("=" * 60)
    print()
    print("📡 服务地址: http://127.0.0.1:5000")
    print("📡 本地访问: http://localhost:5000")
    print()
    print("🔧 已实现接口:")
    print("  POST /api/enroll     - 员工注册（前端使用）")
    print("  POST /api/identify   - 人脸识别（汤艾梧使用）")
    print("  POST /api/attendance - 打卡记录（汤艾梧使用）")
    print("  GET  /api/health     - 健康检查")
    print("  GET  /api/test       - 连通测试")
    print("  GET  /api/employees  - 员工列表")
    print()
    print("🚨 注意:")
    print("  1. 已解决跨域问题，前端可独立运行")
    print("  2. 注册接口格式已严格按照前端要求实现")
    print("  3. 等待集成陈锡翘和黄晨禹的模块")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)