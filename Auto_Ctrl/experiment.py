import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import time
import os
import json
import serial
import threading
import numpy as np
from datetime import datetime
from model import resnet34
from hsv_analyzer import HSVColorAnalyzer
import Find_COM


def _open_best_camera():
    """打开外接摄像头。

    策略：外接摄像头对着滴定实验装置（画面平滑、纹理少），
    笔记本自带摄像头对着用户人脸（面部细节丰富、纹理多）。
    用拉普拉斯方差衡量画面纹理复杂度，选纹理更低的那个。
    """
    best_cap = None
    best_idx = -1
    best_texture = float('inf')

    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue

        frame = None
        for _ in range(6):
            ret, frame = cap.read()

        if not ret or frame is None:
            cap.release()
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()

        if texture < best_texture:
            if best_cap is not None:
                best_cap.release()
            best_cap = cap
            best_idx = idx
            best_texture = texture
        else:
            cap.release()

    return best_cap, best_idx if best_cap is not None else None

# 兰格 GX 蠕动泵 OEM 协议常量
OEM_ADDR = 1
SPEED_EXTRACT = 6000
SPEED_TITRATE = 200
SPEED_DRAIN = 6000
DIR_CW = 1
DIR_CCW = 0
RUN_ON = 0x01
RUN_OFF = 0x00
OEM_BAUDRATE = 9600
OEM_TIMEOUT = 0.3


def _oem_fcs(data: bytes) -> int:
    result = 0
    for b in data:
        result ^= b
    return result & 0xFF


def _oem_wj(ser, speed, run_state, direction, addr=OEM_ADDR):
    pdu = bytes([0x57, 0x4A,
                 (speed >> 8) & 0xFF, speed & 0xFF,
                 run_state & 0xFF,
                 direction & 0xFF])
    raw = bytes([addr, len(pdu)]) + pdu
    fcs = _oem_fcs(raw)
    frame = bytes([0xE9]) + raw + bytes([fcs])
    ser.reset_input_buffer()
    bytes_written = ser.write(frame)
    ser.flush()
    if bytes_written != len(frame):
        raise RuntimeError(
            f"泵写入失败: 期望写入 {len(frame)} 字节, 实际写入 {bytes_written} 字节"
        )
    time.sleep(0.05)


def _oem_rj(ser, addr=OEM_ADDR):
    pdu = bytes([0x52, 0x4A])
    raw = bytes([addr, len(pdu)]) + pdu
    fcs = _oem_fcs(raw)
    frame = bytes([0xE9]) + raw + bytes([fcs])
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    response = ser.read(20)
    if len(response) >= 7 and response[0] == 0xE9:
        resp_len = response[2]
        pdu_start = 3
        pdu_end = pdu_start + resp_len
        if len(response) >= pdu_end + 1:
            pdu_data = response[pdu_start:pdu_end]
            if pdu_data[0:2] == bytes([0x52, 0x4A]) and len(pdu_data) >= 6:
                speed = (pdu_data[2] << 8) | pdu_data[3]
                run_state = pdu_data[4]
                direction = pdu_data[5]
                return speed, run_state, direction
    return None


class TitrationExperiment:
    def __init__(self):
        self.pump_ser = None
        self.ph_ser = None
        self.camera = None

        self.model = None
        self.class_indict = None
        self.data_transform = None
        self.device = None

        self.running = False
        self._purge_running = False
        self.finished = False
        self.total_volume = 0
        self.volume_list = []
        self.voltage_list = []
        self.color_list = []
        self.confidence_list = []
        self.current_color = ""
        self.current_confidence = 0.0
        self.formatted_time = ""

        self.pump_speed_rpm = 2.0
        self.confidence_threshold = 0.5
        self.endpoint_color = "orange"
        self.cycle_interval = 0.5
        self.double_confirm = True
        self.detection_mode = "dl"
        self.hsv_analyzer = None
        self.hsv_confidence_threshold = 0.35
        self.endpoint_streak_frames = 2

        self.on_update = None
        self.on_log = None

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def _log(self, level, message):
        if self.on_log:
            self.on_log(level, message)

    # ---- 硬件连接 ----

    def connect_pump(self):
        serial_ports = Find_COM.list_serial_ports()
        if serial_ports:
            port = serial_ports[0]
            try:
                self.pump_ser = serial.Serial(
                    port=port, baudrate=OEM_BAUDRATE,
                    bytesize=serial.EIGHTBITS, parity=serial.PARITY_EVEN,
                    stopbits=serial.STOPBITS_ONE, timeout=OEM_TIMEOUT
                )
                self._log("info", f"泵串口已打开: {port}")
                # 等待泵硬件初始化完成后再发送命令
                time.sleep(0.2)
                # 重试 RJ 读状态（最多 3 次），避免因时序问题误判为未连接
                result = None
                for attempt in range(3):
                    result = _oem_rj(self.pump_ser)
                    if result is not None:
                        break
                    self._log("warning", f"泵 RJ 命令无响应，重试 {attempt + 1}/3...")
                    time.sleep(0.15)
                if result is not None:
                    speed, run_state, direction = result
                    state_str = "运行中" if (run_state & 0x01) else "已停止"
                    self._log("info", f"泵通讯正常 (速度={speed/100:.2f}rpm, {state_str})")
                    # 确保泵初始为停止状态
                    _oem_wj(self.pump_ser, SPEED_TITRATE, RUN_OFF, DIR_CCW)
                else:
                    self._log("warning", "泵未响应 RJ 命令（已重试 3 次），将使用模拟模式")
                    try:
                        self.pump_ser.close()
                    except Exception:
                        pass
                    self.pump_ser = None
            except Exception as e:
                self._log("error", f"泵连接失败: {e}")
                self.pump_ser = None
        else:
            self._log("warning", "未找到 Serial Port 串口，泵将使用模拟模式")

    def connect_ph_meter(self):
        port_USB = Find_COM.list_USB_ports()
        if port_USB:
            try:
                self.ph_ser = serial.Serial(port_USB[0], baudrate=115200, timeout=1)
                self._log("info", f"pH计已连接: {port_USB[0]}")
            except Exception as e:
                self._log("error", f"pH计连接失败: {e}")
                self.ph_ser = None
        else:
            self._log("warning", "未找到 USB 串口，pH计不可用")

    def list_cameras(self):
        """列出所有可用摄像头及其分辨率"""
        available = []
        for idx in range(5):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append({
                    'index': idx,
                    'width': w,
                    'height': h,
                })
                cap.release()
        return available

    def connect_camera(self, camera_idx=None):
        """连接摄像头，默认通过分辨率检测识别外接摄像头"""
        if camera_idx is not None:
            self.camera = cv2.VideoCapture(camera_idx)
            if not self.camera.isOpened():
                self._log("error", f"无法打开摄像头 (索引 {camera_idx})")
                return False
            w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._log("info", f"摄像头已连接 (索引 {camera_idx}, {w}x{h})")
            return True

        # 自动检测：通过分辨率区分外接与自带
        cap, idx = _open_best_camera()
        if cap is None:
            self._log("error", "未检测到摄像头")
            return False

        self.camera = cap
        w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._log("info", f"摄像头已连接 (索引 {idx}, {w}x{h})")
        return True

    # ---- 模型加载 ----

    def load_model(self, weights_path, class_json_path):
        if self.detection_mode == "hsv":
            self._log("info", "HSV 模式，跳过深度学习模型加载")
            return
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._log("info", f"使用设备: {self.device}")

        json_full = os.path.join(self.base_dir, class_json_path)
        with open(json_full, "r") as f:
            self.class_indict = json.load(f)
        num_classes = len(self.class_indict)
        self._log("info", f"类别: {self.class_indict}")

        weights_full = os.path.join(self.base_dir, weights_path)
        if not os.path.exists(weights_full):
            raise FileNotFoundError(f"权重文件不存在: {weights_full}")

        self.model = resnet34(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(
            torch.load(weights_full, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._log("info", "模型加载完成")

    # ---- 预测 ----

    def predict_image(self, image_path):
        image = Image.open(image_path)
        img = self.data_transform(image)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        color = self.class_indict[str(predict_cla)]
        confidence = float(predict[predict_cla].numpy())
        return color, confidence

    def analyze_color_hsv(self, image_path):
        """使用 HSV 色彩空间分析颜色（无需深度学习模型）。"""
        if self.hsv_analyzer is None:
            self._log("error", "HSV 分析器未初始化")
            return self.endpoint_color, 0.0
        color, confidence = self.hsv_analyzer.analyze_image(image_path)
        return color, confidence

    # ---- 泵控制 ----

    def _speed_to_oem(self, rpm):
        speed = int(rpm * 100)
        # OEM 协议帧以 0xE9 开头，速度值中不能包含 0xE8 或 0xE9 字节
        hi, lo = (speed >> 8) & 0xFF, speed & 0xFF
        if hi in (0xE8, 0xE9) or lo in (0xE8, 0xE9):
            self._log("warning", f"速度 {rpm} rpm 含冲突字节，自动调整 +1")
            speed += 1
        return speed

    def start_pump(self):
        if self.pump_ser is None:
            self._log("info", "(模拟模式) 泵启动跳过")
            return
        speed = self._speed_to_oem(self.pump_speed_rpm)
        _oem_wj(self.pump_ser, speed, RUN_ON, DIR_CCW)
        # 验证泵是否真正启动
        result = _oem_rj(self.pump_ser)
        if result is not None:
            _, run_state, _ = result
            if run_state & 0x01:
                self._log("info", f"滴定泵已启动 ({self.pump_speed_rpm} rpm)")
            else:
                self._log("error", f"泵发送了启动命令但泵未运行! 请检查泵接线和电源")
        else:
            self._log("warning", f"滴定泵启动命令已发送但无法验证状态 ({self.pump_speed_rpm} rpm)")

    def stop_pump(self):
        if self.pump_ser is None:
            return
        try:
            speed = self._speed_to_oem(self.pump_speed_rpm)
            _oem_wj(self.pump_ser, speed, RUN_OFF, DIR_CCW)
            time.sleep(0.1)
            _oem_wj(self.pump_ser, speed, RUN_OFF, DIR_CCW)
            self._log("info", "泵已停止")
        except Exception as e:
            self._log("error", f"泵停止失败: {e}")

    def purge_bubbles(self, direction, duration_sec, on_start=None, on_tick=None, on_done=None):
        """排气泡：高速运转泵指定时长后自动停止。
        direction: 'ccw'（逆时针/排液）或 'cw'（顺时针/吸液）
        duration_sec: 运行秒数
        """
        if self.pump_ser is None:
            if on_done:
                on_done(False, "蠕动泵未连接，无法排气泡")
            return
        if self.running:
            if on_done:
                on_done(False, "实验进行中，请先停止实验再排气泡")
            return
        if self._purge_running:
            if on_done:
                on_done(False, "排气泡已在运行中，请等待当前操作完成")
            return
        self._purge_running = True

        dir_val = DIR_CW if direction == 'cw' else DIR_CCW
        dir_name = '顺时针(吸液)' if direction == 'cw' else '逆时针(排液)'
        self._log("info", f"排气泡: {dir_name}, {duration_sec}秒, 20 rpm")

        try:
            speed = self._speed_to_oem(20)
            # 查询泵初始状态
            init_state = _oem_rj(self.pump_ser)
            if init_state is not None:
                self._log("info", f"泵初始状态: speed={init_state[0]/100:.2f}rpm run={init_state[1]} dir={init_state[2]}")
            # 两步操作：先设速度（停止态），再启动
            _oem_wj(self.pump_ser, speed, RUN_OFF, dir_val)
            time.sleep(0.2)
            _oem_wj(self.pump_ser, speed, RUN_ON, dir_val)
            time.sleep(0.2)
            # 验证泵状态（诊断用，不阻断操作）
            result = _oem_rj(self.pump_ser)
            if result is not None:
                spd, run_state, cur_dir = result
                self._log("info", f"泵当前状态: speed={spd/100:.2f}rpm run={run_state} dir={cur_dir}")
                if run_state & 0x01:
                    self._log("info", "排气泡泵已启动")
                else:
                    self._log("warning", f"泵状态查询返回 run=0（未运行），但已发送启动命令。若泵实际未转动，请检查泵面板最大转速设置")
            else:
                self._log("warning", "无法读取泵状态，但已发送启动命令")
            if on_start:
                on_start(duration_sec, dir_name)
        except Exception as e:
            self._log("error", f"排气泡启动失败: {e}")
            self._purge_running = False
            if on_done:
                on_done(False, str(e))
            return

        def _purge_timer():
            start = time.time()
            while time.time() - start < duration_sec:
                remaining = duration_sec - int(time.time() - start)
                if on_tick:
                    on_tick(remaining)
                time.sleep(0.5)
            try:
                speed = self._speed_to_oem(20)
                _oem_wj(self.pump_ser, speed, RUN_OFF, dir_val)
                time.sleep(0.1)
                _oem_wj(self.pump_ser, speed, RUN_OFF, dir_val)
                self._log("info", "排气泡完成，泵已停止")
                if on_done:
                    on_done(True, "排气泡完成")
            except Exception as e:
                self._log("error", f"排气泡停止失败: {e}")
                if on_done:
                    on_done(False, str(e))
            finally:
                self._purge_running = False

        threading.Thread(target=_purge_timer, daemon=True).start()

    # ---- pH计读数 ----

    def read_voltage(self):
        if self.ph_ser is None:
            return 0.0
        self.ph_ser.write("VOL|\n".encode())
        time.sleep(0.1)
        for _ in range(10):
            response = self.ph_ser.readline().decode().strip()
            if response:
                try:
                    return float(response)
                except ValueError:
                    pass
        return 0.0

    # ---- 实验期间持续读帧线程 ----

    def _capture_loop(self):
        while self.camera is not None and self.camera.isOpened() and self.running:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.05)

    # ---- 主实验循环 ----

    def run_experiment_loop(self, detection_mode=None, hsv_config=None):
        if detection_mode:
            self.detection_mode = detection_mode
        self._log("info", f"检测模式: {self.detection_mode}")

        if self.detection_mode in ("hsv", "hybrid"):
            if hsv_config is None:
                hsv_config = {}
            hsv_config.setdefault("endpoint_color", self.endpoint_color)
            self.hsv_analyzer = HSVColorAnalyzer(hsv_config)
            self._log("info", "HSV 分析器已初始化")

        self.running = True
        self.finished = False
        self.total_volume = 0
        self.volume_list = []
        self.voltage_list = []
        self.color_list = []
        self.confidence_list = []
        self.formatted_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log("info", f"实验开始: {self.formatted_time}")
        if self.double_confirm:
            self._log("info", "终点检测: 双确认模式（连续两次达标才判定终点）")

        self.start_pump()

        cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        cap_thread.start()

        pending_endpoint = False

        try:
            while self.running:
                self.total_volume += 1
                self.volume_list.append(self.total_volume)
                self._log("info", f"循环 #{self.total_volume} 开始")

                with self.frame_lock:
                    frame = self.latest_frame.copy() if self.latest_frame is not None else None

                if frame is None:
                    self._log("warning", "无可用画面，重试...")
                    time.sleep(1)
                    continue

                image_name = f'{self.formatted_time}PH{int(time.time())}.jpg'
                input_dir = os.path.join(self.base_dir, 'Input')
                os.makedirs(input_dir, exist_ok=True)
                filepath = os.path.join(input_dir, image_name)
                success, buf = cv2.imencode('.jpg', frame)
                if success:
                    with open(filepath, 'wb') as f:
                        f.write(buf.tobytes())

                if not os.path.exists(filepath):
                    self._log("warning", f"图片保存失败: {image_name}")
                    continue

                self._log("info", "开始AI识别...")
                if self.detection_mode == "hsv":
                    self.current_color, self.current_confidence = self.analyze_color_hsv(filepath)
                    self._log("info", f"HSV识别结果: {self.current_color} ({self.current_confidence:.3f})")
                elif self.detection_mode == "hybrid":
                    dl_color, dl_conf = self.predict_image(filepath)
                    self._log("info", f"DL识别: {dl_color} ({dl_conf:.3f})")
                    hsv_color, hsv_conf = self.analyze_color_hsv(filepath)
                    self._log("info", f"HSV识别: {hsv_color} ({hsv_conf:.3f})")
                    # 混合模式：DL 为主，两者一致时取 DL 结果，不一致时降低置信度
                    if dl_color == hsv_color:
                        self.current_color = dl_color
                        self.current_confidence = max(dl_conf, hsv_conf)
                    else:
                        self.current_color = dl_color
                        self.current_confidence = dl_conf * 0.7  # 降权
                        self._log("warning", f"DL与HSV结果不一致，降低置信度")
                else:
                    self.current_color, self.current_confidence = self.predict_image(filepath)
                    self._log("info", f"识别结果: {self.current_color} ({self.current_confidence:.3f})")

                self.confidence_list.append(round(self.current_confidence, 4))

                current_voltage = 0.0
                if self.ph_ser is not None:
                    try:
                        current_voltage = self.read_voltage()
                    except Exception:
                        pass
                self.voltage_list.append(current_voltage)

                is_endpoint = (
                    self.current_color == self.endpoint_color
                    and self.current_confidence > self.confidence_threshold
                )

                endpoint_confirmed = False

                if self.double_confirm:
                    if is_endpoint:
                        if pending_endpoint:
                            self.volume_list.pop()
                            self.voltage_list.pop()
                            self.confidence_list.pop()
                            self.total_volume -= 1
                            self.color_list[-1] = 1
                            endpoint_confirmed = True
                            self._log("info", f"视觉终点确认! 体积={self.total_volume}")
                        else:
                            pending_endpoint = True
                            self.color_list.append(0)
                            self._log("info", f"疑似终点 (第1次检测, 置信度={self.current_confidence:.3f}), 等待二次确认...")
                    else:
                        if pending_endpoint:
                            pending_endpoint = False
                            self._log("info", "疑似终点未通过二次确认，继续滴定")
                        self.color_list.append(0)
                else:
                    if is_endpoint:
                        self.color_list.append(1)
                        endpoint_confirmed = True
                        self._log("info", f"视觉终点到达! 体积={self.total_volume}")
                    else:
                        self.color_list.append(0)

                if self.on_update:
                    self.on_update({
                        'volume': self.total_volume,
                        'color': self.current_color,
                        'confidence': round(self.current_confidence, 4),
                        'voltage': round(current_voltage, 4),
                        'is_endpoint': endpoint_confirmed,
                        'pending_endpoint': pending_endpoint and not endpoint_confirmed,
                        'volume_list': self.volume_list,
                        'voltage_list': self.voltage_list,
                        'color_list': self.color_list,
                        'confidence_list': self.confidence_list,
                    })

                if endpoint_confirmed:
                    break

                time.sleep(self.cycle_interval)

        except Exception as e:
            self._log("error", f"实验出错: {e}")
        finally:
            self.stop_pump()
            self.save_results()
            self.finished = True
            self.running = False
            if self.on_update:
                self.on_update({
                    'status': 'finished',
                    'volume_list': self.volume_list,
                    'voltage_list': self.voltage_list,
                    'color_list': self.color_list,
                    'confidence_list': self.confidence_list,
                })
            self._log("info", "实验结束")

    # ---- 结果保存 ----

    def save_results(self):
        if not self.formatted_time or not self.volume_list:
            return
        output = {
            "volume_list": self.volume_list,
            "voltage_list": self.voltage_list,
            "color_list": self.color_list,
            "confidence_list": self.confidence_list,
        }
        out_path = os.path.join(self.base_dir, 'Output', f'{self.formatted_time}.json')
        os.makedirs(os.path.join(self.base_dir, 'Output'), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(output, f)
        self._log("info", f"数据已保存: {out_path}")

    # ---- 停止实验 ----

    def stop(self):
        self.running = False
        self.stop_pump()
        self._log("info", "用户手动停止实验，可开始新实验")

    # ---- 清理资源 ----

    def cleanup(self):
        self.stop_pump()
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None
        if self.pump_ser is not None:
            try:
                self.pump_ser.close()
            except Exception:
                pass
            self.pump_ser = None
        if self.ph_ser is not None:
            try:
                self.ph_ser.close()
            except Exception:
                pass
            self.ph_ser = None
        self._log("info", "所有资源已释放")

    # ---- 关闭系统（完整清理）----

    def shutdown(self):
        self.running = False
        self.stop_pump()
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None
        if self.pump_ser is not None:
            try:
                self.pump_ser.close()
            except Exception:
                pass
            self.pump_ser = None
        if self.ph_ser is not None:
            try:
                self.ph_ser.close()
            except Exception:
                pass
            self.ph_ser = None
        self._log("info", "系统已关闭，所有资源已释放")

    # ---- 空闲摄像头读取（非实验时供视频流使用）----

    def idle_capture_loop(self):
        while self.camera is not None and self.camera.isOpened() and not self.running:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.1)
