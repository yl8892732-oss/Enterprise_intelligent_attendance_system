import cv2
import threading
import time
import queue
from collections import deque
from datetime import datetime
import psutil
import os


class VideoStreamController:
    """视频流控制器：负责摄像头连接、帧捕获和帧率控制"""

    def __init__(self, camera_id=0, target_fps=30, process_every_n=5):
        """
        初始化视频流控制器

        参数:
            camera_id: 摄像头ID (0为默认摄像头)
            target_fps: 目标帧率
            process_every_n: 每N帧处理一次识别，减少计算负担
        """
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.process_every_n = process_every_n
        self.frame_interval = 1.0 / target_fps

        # 视频流对象
        self.cap = None
        self.is_streaming = False

        # 帧队列和计数器
        self.frame_queue = queue.Queue(maxsize=20)  # 限制队列大小防止内存溢出
        self.frame_counter = 0

        # 性能统计
        self.performance_stats = {
            'fps': 0,
            'frame_count': 0,
            'start_time': None
        }

        # 初始化摄像头
        self._init_camera()

    def _init_camera(self):
        """初始化摄像头连接"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"摄像头未开启 {self.camera_id}")

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            print(f"摄像头 {self.camera_id} 初始化成功")
            print(f"分辨率: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"帧率: {self.cap.get(cv2.CAP_PROP_FPS)}")

        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            # 备用方案：使用视频文件测试
            print("尝试加载测试视频...")
            self.cap = cv2.VideoCapture('test_video.mp4')

    def start_streaming(self):
        """开始视频流捕获（多线程）"""
        if self.cap is None:
            print("摄像头未初始化")
            return False

        self.is_streaming = True
        self.performance_stats['start_time'] = time.time()

        # 启动视频流捕获线程
        self.stream_thread = threading.Thread(target=self._capture_frames)
        self.stream_thread.daemon = True
        self.stream_thread.start()

        print("视频流捕获已启动")
        return True

    def _capture_frames(self):
        """视频帧捕获线程函数"""
        last_frame_time = time.time()

        while self.is_streaming and self.cap.isOpened():
            current_time = time.time()
            elapsed = current_time - last_frame_time

            # 控制帧率
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

            # 读取一帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取视频帧")
                time.sleep(0.1)
                continue

            # 更新帧计数器
            self.frame_counter += 1
            self.performance_stats['frame_count'] += 1

            # 计算实时FPS
            if self.performance_stats['frame_count'] % 30 == 0:
                elapsed_total = current_time - self.performance_stats['start_time']
                self.performance_stats['fps'] = self.performance_stats['frame_count'] / elapsed_total

            # 每N帧放入队列供处理
            if self.frame_counter % self.process_every_n == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), self.frame_counter))
                else:
                    # 队列满时丢弃最旧帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((frame.copy(), self.frame_counter))
                    except queue.Empty:
                        pass

            last_frame_time = current_time

    def get_frame_for_processing(self):
        """获取待处理的帧"""
        try:
            frame, frame_num = self.frame_queue.get(timeout=0.1)
            return frame, frame_num
        except queue.Empty:
            return None, 0

    def stop_streaming(self):
        """停止视频流"""
        self.is_streaming = False
        if self.cap:
            self.cap.release()
        print("视频流已停止")

    def get_performance_stats(self):
        """获取性能统计"""
        stats = self.performance_stats.copy()
        stats['queue_size'] = self.frame_queue.qsize()
        stats['memory_usage'] = psutil.Process(os.getpid()).memory_percent()
        return stats


class AntiFraudController:
    """防误判控制器：防止重复打卡和误识别"""

    def __init__(self, cooldown_seconds=30, confirm_frames=5):
        """
        初始化防误判控制器

        参数:
            cooldown_seconds: 冷却时间（秒），同一人两次打卡最小间隔
            confirm_frames: 确认帧数，连续多少帧识别一致才确认打卡
        """
        self.cooldown_seconds = cooldown_seconds
        self.confirm_frames = confirm_frames

        # 打卡记录 {person_id: last_attendance_time}
        self.attendance_records = {}

        # 识别历史 {person_id: deque of recent recognitions}
        self.recognition_history = {}

        # 临时缓冲区，用于连续帧确认
        self.confirmation_buffer = deque(maxlen=confirm_frames)

    def check_can_attendance(self, person_id, person_name, confidence_score, threshold=0.8):
        """
        检查是否可以进行打卡

        参数:
            person_id: 识别的人员ID
            confidence_score: 置信度分数
            threshold: 置信度阈值

        返回:
            (can_attendance, reason)
        """
        current_time = datetime.now()

        # 1. 检查置信度
        if confidence_score < threshold:
            return False, f"置信度过低: {confidence_score:.2f} < {threshold}"

        # 2. 检查冷却时间
        if person_id in self.attendance_records:
            last_time = self.attendance_records[person_id]
            time_diff = (current_time - last_time).total_seconds()

            if time_diff < self.cooldown_seconds:
                remaining = self.cooldown_seconds - time_diff
                return False, f"冷却时间中，请等待 {remaining:.1f} 秒"

        # 3. 连续帧确认机制
        self.confirmation_buffer.append((person_id, person_name, confidence_score))

        if len(self.confirmation_buffer) < self.confirm_frames:
            return False, f"确认中... ({len(self.confirmation_buffer)}/{self.confirm_frames})"

        # 检查缓冲区中的识别结果是否一致
        buffer_ids = [item[0] for item in self.confirmation_buffer]
        buffer_names = [item[1] for item in self.confirmation_buffer]
        buffer_scores = [item[2] for item in self.confirmation_buffer]

        # 所有人脸ID相同且平均置信度高于阈值
        if len(set(buffer_ids)) == 1 and sum(buffer_scores) / len(buffer_scores) >= threshold:
            # 清空缓冲区，准备下一次确认
            self.confirmation_buffer.clear()

            # 更新打卡记录
            self.attendance_records[person_id] = current_time

            # 记录识别历史
            history_key = (person_id, person_name)  # 统一样式
            if history_key not in self.recognition_history:
                self.recognition_history[history_key] = deque(maxlen=100)
            self.recognition_history[history_key].append({
                'time': current_time,
                'score': confidence_score
            })
            return True, f"{person_id}（{person_name}）打卡成功"

        return False, "连续帧识别不一致"

    def get_attendance_status(self, person_id, person_name):
        """获取指定人员的打卡状态"""
        if person_id in self.attendance_records:
            last_time = self.attendance_records[person_id]
            time_diff = (datetime.now() - last_time).total_seconds()
            return {
                'person_name': person_name,
                'last_attendance': last_time.strftime("%Y-%m-%d %H:%M:%S"),
                'seconds_since_last': time_diff,
                'in_cooldown': time_diff < self.cooldown_seconds
            }
        return None


class PerformanceOptimizer:
    """性能优化器：监控和优化系统性能"""

    def __init__(self):
        """初始化性能优化器"""
        self.monitoring = True
        self.performance_data = {
            'frame_processing_times': deque(maxlen=100),
            'recognition_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100)
        }

        # 启动性能监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_performance(self):
        """性能监控线程"""
        while self.monitoring:
            # 监控CPU和内存使用
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory_info = psutil.Process(os.getpid()).memory_info()
            memory_percent = psutil.Process(os.getpid()).memory_percent()

            self.performance_data['cpu_usage'].append(cpu_percent)
            self.performance_data['memory_usage'].append(memory_percent)

            time.sleep(2)  # 每2秒监控一次

    def record_processing_time(self, process_type, time_taken):
        """记录处理时间"""
        if process_type in self.performance_data:
            self.performance_data[process_type].append(time_taken)

    def get_performance_report(self):
        """获取性能报告"""
        report = {}

        for key, data in self.performance_data.items():
            if data:
                report[key] = {
                    'current': data[-1] if data else 0,
                    'avg': sum(data) / len(data) if data else 0,
                    'max': max(data) if data else 0,
                    'min': min(data) if data else 0,
                    'count': len(data)
                }

        # 建议优化策略
        report['optimization_suggestions'] = self._generate_suggestions(report)

        return report

    def _generate_suggestions(self, report):
        """生成优化建议"""
        suggestions = []

        # 检查帧处理时间
        if 'frame_processing_times' in report:
            avg_time = report['frame_processing_times']['avg']
            if avg_time > 0.1:  # 如果平均处理时间超过100ms
                suggestions.append("帧处理时间过长，建议减少处理频率或优化算法")

        # 检查内存使用
        if 'memory_usage' in report:
            avg_memory = report['memory_usage']['avg']
            if avg_memory > 70:  # 如果内存使用超过70%
                suggestions.append("内存使用过高，建议清理缓存或优化内存管理")

        # 检查CPU使用
        if 'cpu_usage' in report:
            avg_cpu = report['cpu_usage']['avg']
            if avg_cpu > 80:  # 如果CPU使用超过80%
                suggestions.append("CPU使用过高，建议启用多线程或降低处理频率")

        return suggestions if suggestions else ["系统运行良好，无需优化"]


# 主控制器类：整合所有功能
class AttendanceFlowController:
    """考勤流程主控制器"""

    def __init__(self):
        """初始化主控制器"""
        # 初始化各模块
        self.video_controller = VideoStreamController(
            camera_id=0,
            target_fps=30,
            process_every_n=5  # 每5帧处理一次
        )

        self.anti_fraud = AntiFraudController(
            cooldown_seconds=30,  # 30秒冷却时间
            confirm_frames=5  # 连续5帧确认
        )

        self.performance_optimizer = PerformanceOptimizer()

        # 状态变量
        self.is_running = False
        self.last_processed_frame = 0

        # 识别结果回调函数（由其他模块设置）
        self.recognition_callback = None

    def set_recognition_callback(self, callback_func):
        """设置识别结果回调函数"""  ###回调人脸识别函数
        self.recognition_callback = callback_func

    def start(self):
        """启动考勤流程"""
        if self.is_running:
            print("系统已经在运行中")
            return False

        # 启动视频流
        if not self.video_controller.start_streaming():
            print("无法启动视频流")
            return False

        self.is_running = True
        # 改为线程启动
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        print("考勤主循环已在后台启动")
        return True

    def _main_loop(self):
        """主处理循环"""
        while self.is_running:
            try:
                # 1. 获取待处理帧
                frame, frame_num = self.video_controller.get_frame_for_processing()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 记录处理开始时间
                process_start = time.time()

                # 2. 如果有识别回调，调用人脸识别模块
                if self.recognition_callback:
                    recognition_result = self.recognition_callback(frame)

                    # 3. 如果有识别结果，进行防误判检查
                    if recognition_result and 'person_id' in recognition_result:
                        person_id = recognition_result.get('person_id')
                        person_name = recognition_result.get('person_name', 'none')
                        confidence = recognition_result.get('confidence', 0)

                        # 防误判检查
                        can_attend, reason = self.anti_fraud.check_can_attendance(
                            person_id, person_name, confidence
                        )

                        if can_attend:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}]  {person_id}（{person_name}） 打卡成功")
                            # 这里可以触发打卡记录保存等操作
                        else:
                            # 显示原因（调试用）
                            if "确认中" not in reason:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]  {person_id}({person_name}): {reason}")

                # 记录处理时间
                process_time = time.time() - process_start
                self.performance_optimizer.record_processing_time('frame_processing_times', process_time)

                # 控制循环频率
                time.sleep(0.01)

            except KeyboardInterrupt:
                print("接收到中断信号")
                self.stop()
                break
            except Exception as e:
                print(f"主循环错误: {e}")
                time.sleep(0.1)

    def stop(self):
        """停止系统"""
        self.is_running = False
        self.video_controller.stop_streaming()
        print("考勤系统已停止")

    def get_system_status(self):
        """获取系统状态"""
        video_stats = self.video_controller.get_performance_stats()
        perf_report = self.performance_optimizer.get_performance_report()

        status = {
            'system_running': self.is_running,
            'video_stats': video_stats,
            'performance_report': perf_report,
            'anti_fraud_status': {
                'cooldown_seconds': self.anti_fraud.cooldown_seconds,
                'confirm_frames': self.anti_fraud.confirm_frames,
                'total_attendance': len(self.anti_fraud.attendance_records)
            }
        }

        return status