"""
修复版FaceProcessor - 解决检测逻辑问题
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO

class FaceProcessor:
    def __init__(self, target_size=112):
        self.target_size = target_size
        self.model = YOLO("yolov8n.pt")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print(f"✅ FaceProcessor初始化完成")

    def detect_faces(self, image):
        """修复版人脸检测 - 基于诊断结果"""

        # YOLO人物检测（已验证正常）
        results = self.model(image, verbose=False)

        all_faces = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # 人物检测（基于你的诊断结果：置信度0.8-0.9）
                    if cls == 0 and conf > 0.2:  # 降低阈值，基于你的0.8+结果
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # 在人物区域内检测人脸（已验证正常）
                        person_roi = image[y1:y2, x1:x2]
                        if person_roi.size == 0:
                            continue

                        # 人脸检测（基于你的诊断结果：每张图1张人脸）
                        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                        face_rects = self.face_cascade.detectMultiScale(
                            gray_roi,
                            scaleFactor=1.05,  # 更精细（你的诊断显示1.05有效）
                            minNeighbors=3,    # 更宽松（你的诊断显示3有效）
                            minSize=(20, 20)   # 更小（你的诊断显示20有效）
                        )

                        # 关键修复：正确处理每个检测到的人脸
                        for fx, fy, fw, fh in face_rects:
                            # 转换回全图坐标
                            face_x1 = x1 + fx
                            face_y1 = y1 + fy
                            face_x2 = face_x1 + fw
                            face_y2 = face_y1 + fh

                            # 构造关键点
                            landmarks = [
                                (face_x1 + fw//4, face_y1 + fh//4),
                                (face_x1 + 3*fw//4, face_y1 + fh//4),
                                (face_x1 + fw//2, face_y1 + fh//2),
                                (face_x1 + fw//4, face_y1 + 3*fh//4),
                                (face_x1 + 3*fw//4, face_y1 + 3*fh//4)
                            ]

                            face_info = {
                                'box': (face_x1, face_y1, face_x2, face_y2),
                                'landmarks': landmarks,
                                'confidence': conf * 0.9,  # 基于你的高置信度
                                'source': 'yolo+opencv'
                            }
                            all_faces.append(face_info)

        return all_faces

    def align_face(self, image, landmarks):
        """人脸对齐"""
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        desired_eye_distance = 60
        current_eye_distance = np.sqrt(dx**2 + dy**2)
        scale = desired_eye_distance / current_eye_distance if current_eye_distance > 0 else 1.0

        eyes_center = (float(left_eye[0] + right_eye[0]) / 2,
                       float(left_eye[1] + right_eye[1]) / 2)

        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        rotation_matrix[0, 2] += self.target_size // 2 - eyes_center[0]
        rotation_matrix[1, 2] += self.target_size // 3 - eyes_center[1]

        aligned_face = cv2.warpAffine(
            image, rotation_matrix,
            (self.target_size, self.target_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return aligned_face

    def enhance_face(self, face_img):
        """人脸增强"""
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ycrcb[:, :, 0])
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return enhanced

    def process_image(self, image_path):
        """主处理流程"""
        import numpy as np
        file_bytes = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 检测人脸（使用修复版检测）
        faces = self.detect_faces(image)
        print(f"检测到 {len(faces)} 张人脸")

        results = []
        for i, face_data in enumerate(faces):
            x1, y1, x2, y2 = face_data['box']
            landmarks = face_data['landmarks']

            # 裁剪并对齐人脸
            aligned_face = self.align_face(image, landmarks)

            # 增强处理
            enhanced_face = self.enhance_face(aligned_face)

            result = {
                'face_img': enhanced_face,  # 112x112 BGR
                'box': (x1, y1, x2, y2),
                'confidence': face_data['confidence'],
                'person_id': f"face_{i+1}"
            }
            results.append(result)

        return results

    def batch_process_folder(self, input_folder, output_folder):
        """批量处理文件夹"""
        os.makedirs(output_folder, exist_ok=True)

        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        image_files = []

        for file in os.listdir(input_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_folder, file))

        print(f"📁 找到 {len(image_files)} 张图片")

        for image_file in image_files:
            try:
                results = self.process_image(image_file)

                base_name = os.path.splitext(os.path.basename(image_file))[0]

                for i, result in enumerate(results):
                    output_path = os.path.join(output_folder, f"{base_name}_face_{i + 1}.jpg")
                    cv2.imwrite(output_path, result['face_img'])

                    coord_path = os.path.join(output_folder, f"{base_name}_face_{i + 1}_coords.txt")
                    with open(coord_path, 'w') as f:
                        f.write(f"box: {result['box']}\n")
                        f.write(f"confidence: {result['confidence']}\n")

                print(f"✅ 处理完成: {os.path.basename(image_file)} -> {len(results)} 张人脸")

            except Exception as e:
                print(f"❌ 处理失败: {image_file} - {e}")

    def final_output(self, input_folder='test_images', output_folder='output_faces'):
        """最终输出 - 专用文件夹 + 固定格式"""

        print("🚀 最终输出到专用文件夹")
        print(f"输入: {input_folder}")
        print(f"输出: {output_folder}")

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 使用项目根目录（100%正确）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_abs = os.path.join(project_root, input_folder)
        output_abs = os.path.join(project_root, output_folder)

        print(f"绝对路径输入: {input_abs}")
        print(f"绝对路径输出: {output_abs}")

        # 检查输入文件夹是否存在
        if not os.path.exists(input_abs):
            print(f"❌ 输入文件夹不存在: {input_abs}")
            return

        # 获取所有图片文件
        image_files = []
        for file in os.listdir(input_abs):
            if any(ext in file.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']):
                image_files.append(os.path.join(input_abs, file))

        print(f"📁 找到 {len(image_files)} 张图片")

        # 处理每张图片
        for image_file in image_files:
            try:
                results = self.process_image(image_file)

                base_name = os.path.splitext(os.path.basename(image_file))[0]

                for i, result in enumerate(results):
                    # 输出到指定文件夹
                    face_output = os.path.join(output_abs, f"{base_name}_face_{i + 1}.jpg")
                    cv2.imwrite(face_output, result['face_img'])

                    # 输出坐标信息到指定文件夹
                    coord_output = os.path.join(output_abs, f"{base_name}_face_{i + 1}_coords.txt")
                    with open(coord_output, 'w') as f:
                        f.write(f"box: {result['box']}\n")
                        f.write(f"confidence: {result['confidence']}\n")
                        f.write(f"shape: (112, 112, 3)\n")
                        f.write(f"format: BGR\n")

                    print(f"✅ 输出: {face_output}")
                    print(f"✅ 输出: {coord_output}")

            except Exception as e:
                print(f"❌ 处理失败: {image_file} - {e}")

        print("\n✅ 最终输出完成！")



if __name__ == "__main__":
    print("🚀 直接运行face_processor最终输出")
    print("=" * 50)

    processor = FaceProcessor(target_size=112)

    # 直接运行最终输出到专用文件夹
    processor.final_output(input_folder=r'D:\陈锡翘\face_yolo_project\test_images',
                           output_folder=r'D:\陈锡翘\face_yolo_project\output_faces')

    print("\n✅ 直接运行完成！")