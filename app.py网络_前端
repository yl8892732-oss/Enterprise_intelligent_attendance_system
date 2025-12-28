import traceback
import os
import time
import ssl
import cv2
import numpy as np
import base64
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, Response, jsonify

# ===================== 1. 模块导入 =====================
from models import ArcFaceModel
from utils import set_memory_growth
from database import db  # 盘先琰 (PXY)
from face_recognizer import recognizer  # 盘先琰 (PXY)
from project import AntiFraudController  # 汤艾梧 (TAW)
# 注意：report 模块里必须加上 matplotlib.use('Agg')
from report import read_attendance_data, analyze_attendance, visualize_attendance
from cxq_module import CXQFaceProcessor

# ===================== 2. 环境配置 =====================
# 解决 HTTPS 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# 适配 Mac M4 字体
plt.rcParams["font.sans-serif"] = ["PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# 显存优化
set_memory_growth()

# ===================== 3. 核心组件初始化 =====================

# ⚠️ 删除了 VideoStreamController，因为现在由前端控制摄像头

# 防误判控制器：
# cooldown_seconds=30: 打卡成功后 30秒内不能重复打卡
# confirm_frames=1: Web端每次只传一张图，所以设为 1 即可
anti_fraud = AntiFraudController(cooldown_seconds=30, confirm_frames=1)

# 初始化 HCY 的 ArcFace 模型
hcy_model = ArcFaceModel(size=112, backbone_type='ResNet50', training=False, embd_shape=512)
hcy_model.load_weights('hcy_weights.weights.h5')
print("🚀 [系统启动] Web端考勤服务已就绪 (MacBook M4 Optimized)")

# 初始化 Flask
app = Flask(__name__)
# 确保上传文件夹存在
os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
os.makedirs('data', exist_ok=True)


# ===================== 4. 辅助函数 =====================

def extract_feature(face_img):
    """提取人脸特征向量 (适配 HCY 模型)"""
    # 1. 尺寸缩放
    face_img = cv2.resize(face_img, (112, 112))
    # 2. 归一化
    face_img = face_img.astype(np.float32)
    face_img = (face_img - 127.5) / 128.0
    # 3. 维度扩展
    face_img = np.expand_dims(face_img, axis=0)
    # 4. 推理
    embedding = hcy_model.predict(face_img)
    # 5. L2 归一化
    return embedding / np.linalg.norm(embedding)


# 注入比对器
recognizer.set_extract_feature_function(lambda img: extract_feature(img))


def base64_to_cv2(base64_string):
    """将前端传来的 Base64 字符串转为 OpenCV 图片"""
    try:
        if "," in base64_string:
            header, encoded = base64_string.split(",", 1)
        else:
            encoded = base64_string

        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ 图片解码失败: {e}")
        return None


# ===================== 5. 路由逻辑 =====================

@app.route('/')
def index():
    """首页：加载包含 JS 摄像头的打卡页面"""
    return render_template('index.html')


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """Web端打卡接口：接收 Base64 图片 -> 返回识别结果"""
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'status': 'error', 'msg': '未接收到图片'})

        # 1. 解码图片
        frame = base64_to_cv2(image_data)
        if frame is None:
            return jsonify({'status': 'error', 'msg': '图片解码失败'})

        # 2. 识别逻辑
        feat = extract_feature(frame)
        emp_id, sim, recognized, info = recognizer.identify(feat.flatten().tolist())

        # 3. 结果处理
        if recognized:
            name = info.get('name', 'Unknown')
            # 检查冷却时间
            can_attend, msg = anti_fraud.check_can_attendance(emp_id, name, float(sim))

            if can_attend:
                # 写入数据库
                db.add_attendance_record(emp_id, float(sim))
                print(f"✅ [API打卡] 成功: {name}")
                return jsonify({
                    'status': 'success',
                    'name': name,
                    'sim': float(sim),
                    'msg': '打卡成功'
                })
            else:
                # 冷却期内
                return jsonify({
                    'status': 'cool_down',
                    'name': name,
                    'msg': '刚刚已打卡'
                })
        else:
            return jsonify({'status': 'unknown', 'msg': '未识别到员工'})

    except Exception as e:
        print(f"❌ 识别接口报错: {e}")
        return jsonify({'status': 'error', 'msg': str(e)})


@app.route('/register', methods=['GET'])
def register_page():
    """注册页面 (GET)"""
    # 注意：这里的 register.html 也需要修改为使用 JS 调用摄像头
    # 如果你还没改 register.html，它暂时无法拍照
    return render_template('register.html')


@app.route('/api/register', methods=['POST'])
def api_register():
    """Web端注册接口：接收 Base64 + 姓名 + 工号"""
    try:
        data = request.json
        name = data.get('name')
        emp_id = data.get('emp_id')
        image_data = data.get('image')

        if not (name and emp_id and image_data):
            return jsonify({'status': 'error', 'msg': '信息不完整'})

        print(f"🔍 [API注册] 正在处理: {name} ({emp_id})")

        # 1. 解码图片
        frame = base64_to_cv2(image_data)
        if frame is None:
            return jsonify({'status': 'error', 'msg': '图片无效'})

        # 2. 提取特征
        feat = extract_feature(frame)
        vector_list = feat.flatten().tolist()

        # 3. 存入数据库
        if db.add_employee(emp_id, name):
            db.add_face_template(emp_id, vector_list)
            recognizer.refresh_templates()  # 刷新内存

            # 4. 保存照片备份
            save_path = os.path.join('static', 'uploads', f"{emp_id}_{name}.jpg")
            cv2.imwrite(save_path, frame)

            print(f"✅ [注册成功] {name} 已入库")
            return jsonify({'status': 'success', 'msg': f'员工 {name} 注册成功！'})
        else:
            return jsonify({'status': 'fail', 'msg': '该工号已存在'})

    except Exception as e:
        print(f"❌ 注册接口报错: {e}")
        return jsonify({'status': 'error', 'msg': str(e)})


@app.route('/report')
def report():
    """考勤报表 (集成 Matplotlib 修复版)"""
    records = []
    try:
        print("📊 正在生成报表...")
        # 1. 读数据
        df = read_attendance_data(db)

        if not df.empty:
            # 2. 分析 (计算迟到/正常)
            df = analyze_attendance(df)

            # 3. 画图 (注意 report.py 需开启 Agg 模式)
            visualize_attendance(df)

            # 4. 格式化数据传给前端
            if 'check_time' in df.columns:
                df['check_time'] = df['check_time'].astype(str)

            # 转换为字典列表
            records = df.to_dict(orient='records')
        else:
            print("⚠️ 暂无数据")

    except Exception as e:
        print(f"❌ 报表生成失败: {e}")
        traceback.print_exc()
        records = []

    return render_template('report.html', records=records)


@app.route('/history')
def history():
    """查看历史注册照片"""
    upload_dir = os.path.join('static', 'uploads')
    if not os.path.exists(upload_dir):
        return render_template('history.html', records=[])

    all_files = os.listdir(upload_dir)
    image_files = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    return render_template('history.html', records=image_files)


if __name__ == '__main__':
    # 0.0.0.0 允许局域网访问
    app.run(host='0.0.0.0', port=5001, debug=True, ssl_context='adhoc')
