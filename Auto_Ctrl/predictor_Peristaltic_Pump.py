import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import time
import os
from model import resnet34
import json
import serial
from datetime import datetime
from scipy.optimize import curve_fit
import numpy as np
import re
import Find_COM

def _open_best_camera():
    """打开外接摄像头。

    策略：外接摄像头对着滴定实验装置（画面平滑、纹理少），
    笔记本自带摄像头对着用户人脸（面部细节丰富、纹理多）。
    用拉普拉斯方差衡量画面纹理复杂度，选纹理更低的那个。
    不依赖索引、名称、分辨率。
    """
    best_cap = None
    best_idx = -1
    best_texture = float('inf')

    for idx in [0, 1, 2, 3, 4, 5]:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue

        # 读取多帧让摄像头稳定
        frame = None
        for _ in range(6):
            ret, frame = cap.read()

        if not ret or frame is None:
            cap.release()
            continue

        # 拉普拉斯方差衡量纹理复杂度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"  扫描摄像头 [{idx}]: 纹理={texture:.0f}")

        if texture < best_texture:
            # 纹理更低 → 更可能是实验场景 → 外接摄像头
            if best_cap is not None:
                best_cap.release()
            best_cap = cap
            best_idx = idx
            best_texture = texture
        else:
            cap.release()

    if best_cap is not None:
        w = int(best_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(best_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"外接摄像头已连接 (索引 {best_idx}, {w}x{h}, 纹理={best_texture:.0f})")
    else:
        print("未检测到任何摄像头")

    return best_cap

# ============================================================
# 兰格 GX 系列蠕动泵 — 兰格 OEM 协议 通讯配置
# ============================================================

OEM_ADDR = 1        # 泵地址（默认1，如改过需在泵面板查看）

# 速度常量（OEM 协议单位: 0.01 rpm，6000 = 60.00 rpm）
# G100-1L 最大 10000，G300-1L 最大 30000，G600-1L 最大 60000
# ⚠ 速度值不能含 0xE8 或 0xE9 字节（如 232=0x00E8, 1000=0x03E8 会导致泵不响应）
SPEED_EXTRACT = 6000    # 抽取液体: 60.00 rpm
SPEED_TITRATE = 200     # 滴定:     2.00 rpm（0x00C8，安全值）
SPEED_DRAIN   = 6000    # 排液:    60.00 rpm

# 方向 + 运行状态常量
DIR_CW  = 1   # 顺时针
DIR_CCW = 0   # 逆时针
RUN_ON  = 0x01   # 启动 (bit0=1)
RUN_OFF = 0x00   # 停止 (bit0=0)
FULL_SPEED = 0x02  # 全速 (bit1=1)

# 串口参数（必须与泵面板设置一致）
OEM_BAUDRATE = 9600
OEM_TIMEOUT  = 0.3   # 读取超时（秒）


def _oem_fcs(data: bytes) -> int:
    """兰格 OEM 协议 FCS: 逐字节异或"""
    result = 0
    for b in data:
        result ^= b
    return result & 0xFF


def _oem_wj(ser, speed, run_state, direction, addr=OEM_ADDR):
    """OEM WJ 命令: 设置运行参数（速度 + 启停 + 方向）
       PDU: 57 4A + speed(2B BE) + run_state(1B) + direction(1B)
       注意: speed 值不能含 0xE8 或 0xE9 字节"""
    pdu = bytes([0x57, 0x4A,
                 (speed >> 8) & 0xFF, speed & 0xFF,
                 run_state & 0xFF,
                 direction & 0xFF])
    raw = bytes([addr, len(pdu)]) + pdu
    fcs = _oem_fcs(raw)
    frame = bytes([0xE9]) + raw + bytes([fcs])
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    time.sleep(0.05)


def _oem_rj(ser, addr=OEM_ADDR):
    """OEM RJ 命令: 读取泵状态，返回 (speed, run_state, direction) 或 None"""
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


def _check_pump(ser):
    """检查泵通讯是否正常（OEM RJ 读状态）"""
    try:
        result = _oem_rj(ser)
        if result is not None:
            speed, state, direction = result
            state_str = "运行中" if (state & 0x01) else "已停止"
            dir_str = "顺时针" if (direction & 0x01) else "逆时针"
            print(f"[OEM] 泵通讯正常 (速度={speed/100:.2f}rpm, {state_str}, {dir_str})")
            return True
        else:
            print("[OEM] 警告: 泵未响应，请检查泵电源和RS485接线")
            return False
    except Exception as e:
        print(f"[OEM] 警告: 泵通讯检测失败: {e}")
        return False


def _safe_stop_pump(ser):
    """安全停止泵（只发停止命令，不关闭串口）"""
    if ser is None:
        return
    try:
        _oem_wj(ser, SPEED_TITRATE, RUN_OFF, DIR_CCW)
        time.sleep(0.1)
        # 再发一次确保收到（RS485 半双工可能丢帧）
        _oem_wj(ser, SPEED_TITRATE, RUN_OFF, DIR_CCW)
        print("[安全] 泵已停止")
    except Exception as e:
        print(f"[安全] 泵停止失败: {e}")


def get_picture(frame, typ=0, date=''):
    """拍照并保存图片"""
    if frame is None:
        print("错误: 摄像头未捕获到画面")
        return None
    if typ:
        image_name = f'{date}{int(time.time())}.jpg'
    else:
        image_name = f'{date}PH{int(time.time())}.jpg'
    filepath = "Input/" + image_name
    os.makedirs("Input", exist_ok=True)
    success, buf = cv2.imencode('.jpg', frame)
    if success:
        with open(filepath, 'wb') as f:
            f.write(buf.tobytes())
    else:
        return None
    return image_name


def start_move_1(ser):
    """蠕动泵动作1: 抽取液体（顺时针快速运行30秒后停止）"""
    if ser is None:
        return
    _oem_wj(ser, SPEED_EXTRACT, RUN_ON, DIR_CW)
    time.sleep(30)
    _oem_wj(ser, SPEED_EXTRACT, RUN_OFF, DIR_CW)
    print('完成抽取')


def start_move_2(ser):
    """蠕动泵动作2: 开始滴定（逆时针低速持续运行直到终点）"""
    if ser is None:
        print("（模拟模式: 串口操作跳过）")
        return
    _oem_wj(ser, SPEED_TITRATE, RUN_ON, DIR_CCW)
    time.sleep(0.1)
    print('滴定泵已启动，等待终点...')


def start_move_4(ser):
    """蠕动泵动作4: 快速排液（顺时针快速运行）"""
    if ser is None:
        return
    _oem_wj(ser, SPEED_DRAIN, RUN_ON, DIR_CW)
    time.sleep(0.1)
    print('快速排液已启动')


def poly_func(x, a, b, c, d):
    """双曲正切拟合函数，用于滴定曲线拟合"""
    return a * np.tanh(d * x + b) + c


def line_chart(date="1", volume_list=[], voltage_list=[], color_list=[]):
    """绘制滴定曲线图（电位曲线 + 颜色曲线 + 二阶导数）"""
    x = volume_list
    y = voltage_list
    z = color_list

    fig, ax1 = plt.subplots()
    plt.title("titration curve")

    color = 'tab:red'
    ax1.set_xlabel('value')
    ax1.set_ylabel('voltage', color=color)
    ax1.plot(x, y, color=color, antialiased=True)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('color', color=color)
    ax2.plot(x, z, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['yellow', 'orange'])
    ax2.spines['right'].set_position(('outward', 60))

    try:
        popt, pcov = curve_fit(poly_func, x, y, p0=[max(y)*3/4, -max(x), max(y), 1.5])
        print("最优参数:", popt)
        print(f'电位突跃点：{-popt[1]/popt[3]:.3f}')

        x_d = np.arange(0, max(x), 0.05)
        y_fit = poly_func(x_d, *popt)
        dE_dV = np.gradient(y_fit)
        d2E_dV2 = np.gradient(dE_dV)
        y2 = d2E_dV2.tolist()

        ax3 = ax1.twinx()
        color = 'tab:green'
        ax3.set_ylabel('2nd Derivative', color=color)
        ax3.plot(x_d, y2, color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.grid(True, linestyle='--', linewidth=0.5, color='gray', axis='both')

        x_d, y_d = -popt[1]/popt[3], 0.0
        ax3.plot(x_d, y_d, 'ro')
        ax3.annotate(f'({x_d:.2f})', xy=(x_d, y_d), color='red', xytext=(x_d-1, y_d + max(y2)/10))

        visual_idx = next((i for i, v in enumerate(color_list) if v == 1), None)
        if visual_idx is not None and visual_idx < len(x):
            x_c, y_c = x[visual_idx] - 0.025, 0.0
            ax3.plot(x_c, y_c, 'bo')
            ax3.annotate(f'({x_c:.2f})', xy=(x_c, y_c), color='blue', xytext=(x_c - 1, y_c - max(y2) / 10))
            print(f"视觉突跃点：{x_c:.3f}")
    except Exception as e:
        print(e)
        pass

    fig.tight_layout()
    plt.savefig(f'Output/{date}.png')
    plt.show()
    plt.pause(1)
    plt.close()


def predictor(im_file, model, class_indict, data_transform, device):
    """预测单张图片的分类（模型从外部传入，避免重复加载）"""
    image = Image.open(im_file)
    img = data_transform(image)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    class_a = "{}".format(class_indict[str(predict_cla)])
    prob_b = float(predict[predict_cla].numpy())
    print(f"识别结果: {class_a}, 概率: {prob_b:.3f}")
    return class_a, prob_b


def voltage(ser):
    """读取 pH 计电压值"""
    ser.write("VOL|\n".encode())
    time.sleep(0.1)
    for _ in range(10):
        response = ser.readline().decode().strip()
        if response:
            try:
                return float(response)
            except ValueError:
                pass
    return 0.0


def main():
    # 1. 查找 Serial Port 串口（蠕动泵），使用 Modbus RTU 8E1 参数
    serial_ports = Find_COM.list_serial_ports()
    if serial_ports:
        port = serial_ports[0]  # 根据你的电脑调整索引，一般是 [0]
        print(f"找到 Serial Port: {port}")
        pump_ser = serial.Serial(
            port=port,
            baudrate=OEM_BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_ONE,
            timeout=OEM_TIMEOUT
        )
        print(f"已连接 {port}，波特率 {OEM_BAUDRATE}，8E1 (兰格 OEM 协议)")
        # 验证泵通讯，确保泵初始为停止状态
        _check_pump(pump_ser)
        _safe_stop_pump(pump_ser)
    else:
        print("警告: 未找到 Serial Port 串口设备，将使用模拟模式（串口操作将被跳过）")
        pump_ser = None

    # 2. 查找 USB 串口（pH 计）
    USB_ser = None
    port_USB = Find_COM.list_USB_ports()
    if port_USB:
        try:
            USB_ser = serial.Serial(port_USB[0], baudrate=115200, timeout=1)
            print(f"找到 USB 串口: {port_USB[0]}")
        except Exception as e:
            print(f"USB 串口连接失败: {e}")

    # 3. 自动检测并打开摄像头（通过分辨率区分外接与自带）
    cap = _open_best_camera()
    if cap is None:
        print("错误: 未检测到可用摄像头！请检查摄像头连接。")
        if pump_ser is not None:
            pump_ser.close()
        return

    # 4. 加载 AI 模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    json_path = './class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    print(f"类别信息: {class_indict}")

    weights_path = "./resnet34-1Net.pth"
    assert os.path.exists(weights_path), f"错误: 权重文件 '{weights_path}' 不存在。"
    model = resnet34(num_classes=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print("模型加载完成。")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. 开始实验
    total_volume = 0
    volume_list = []
    voltage_list = []
    color_list = []

    start_time = time.time()
    dt_object = datetime.fromtimestamp(start_time)
    formatted_time = dt_object.strftime('%Y%m%d_%H%M%S')
    print("实验开始于", formatted_time)

    try:
        if pump_ser is not None:
            start_move_2(pump_ser)
        else:
            print("（模拟模式: 串口操作跳过）")

        while True:
            total_volume += 1
            volume_list.append(total_volume)

            ret, frame = cap.read()
            if not ret or frame is None:
                print("警告: 无法从摄像头读取画面，重试...")
                time.sleep(1)
                continue

            name = get_picture(frame, 0, formatted_time)
            if name is None:
                continue
            im_file = 'Input/' + name

            cv2.imshow('Color', frame)
            cv2.waitKey(1)

            class_a, prob_b = predictor(im_file, model, class_indict, data_transform, device)

            if USB_ser is not None:
                try:
                    voltage_list.append(voltage(USB_ser))
                except Exception:
                    voltage_list.append(0)

            if class_a == "orange" and prob_b > 0.5:
                if pump_ser is not None:
                    _safe_stop_pump(pump_ser)
                    try:
                        pump_ser.close()
                    except Exception:
                        pass
                    pump_ser = None
                print('----->> 视觉终点到达！<<-----')
                print(f"终点体积: {total_volume}")
                print(f"终点照片: {im_file}")
                color_list.append(1)
                break

            color_list.append(0)
            print(f"当前体积: {total_volume}")

    except KeyboardInterrupt:
        print("\n[中断] 用户手动停止实验")
        _safe_stop_pump(pump_ser)
    finally:
        print(f"体积列表: {volume_list}")
        print(f"电压列表: {voltage_list}")
        print(f"颜色列表: {color_list}")

        if len(volume_list) > 0:
            with open(f'Output/{formatted_time}.json', 'w') as f:
                json.dump({
                    "volume_list": volume_list,
                    "voltage_list": voltage_list,
                    "color_list": color_list
                }, f)

        cap.release()
        cv2.destroyAllWindows()
        if pump_ser is not None:
            try:
                _safe_stop_pump(pump_ser)
                pump_ser.close()
            except Exception:
                pass
        if USB_ser is not None:
            try:
                USB_ser.close()
            except Exception:
                pass

        if USB_ser is not None and len(voltage_list) > 0:
            line_chart(formatted_time, volume_list=volume_list, voltage_list=voltage_list, color_list=color_list)

        print("实验结束，所有资源已释放。")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
