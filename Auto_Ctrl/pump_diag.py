#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
泵通讯诊断工具 —— 测试兰格 GX 系列蠕动泵是否正常响应
支持 Modbus RTU 和兰格 OEM 两种协议自动检测
"""
import serial
import time
import Find_COM


# ============================================================
# CRC / FCS 计算
# ============================================================

def crc16_modbus(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def xor_fcs(data: bytes) -> int:
    """兰格 OEM 协议 FCS: 逐字节异或"""
    result = 0
    for b in data:
        result ^= b
    return result & 0xFF


# ============================================================
# Modbus RTU 测试
# ============================================================

def test_modbus(ser, slave=1):
    """尝试用 Modbus RTU 协议与泵通讯"""
    print("\n" + "="*50)
    print("【测试1】Modbus RTU 协议 (地址=%d)" % slave)
    print("="*50)

    # 尝试读多个已知的寄存器
    registers_to_try = [
        (0x0000, "速度"),
        (0x0002, "启停状态"),
        (0x0003, "方向"),
        (0x0020, "上电状态"),
        (0x0040, "加速度"),
        (0x0001, "全速状态"),
    ]

    for addr, name in registers_to_try:
        frame = bytes([slave, 0x03,
                       (addr >> 8) & 0xFF, addr & 0xFF,
                       0x00, 0x01])
        frame += crc16_modbus(frame)

        ser.reset_input_buffer()
        ser.write(frame)
        ser.flush()
        time.sleep(0.1)

        # 尝试读取响应
        response = ser.read(20)
        if len(response) > 0:
            print(f"  [{name}(0x{addr:04X})] 收到 {len(response)} 字节: {response.hex(' ')}")
            if len(response) >= 5 and response[1] == 0x03:
                byte_count = response[2]
                if len(response) >= 3 + byte_count + 2:
                    data = response[3:3 + byte_count]
                    val = int.from_bytes(data, 'big') if len(data) <= 4 else None
                    print(f"    → 寄存器值 = {val} (0x{data.hex()})")
                    return True
            elif len(response) >= 5 and response[1] == 0x83:
                print(f"    → Modbus 异常码 0x{response[2]:02X} (功能码不支持或地址无效)")
        else:
            print(f"  [{name}(0x{addr:04X})] 无响应")

    print("  Modbus RTU: 所有寄存器均无响应")
    return False


# ============================================================
# 兰格 OEM 协议测试
# ============================================================

def test_oem_protocol(ser, addr=1):
    """尝试用兰格 OEM 协议与泵通讯"""
    print("\n" + "="*50)
    print("【测试2】兰格 OEM 协议 (地址=%d)" % addr)
    print("="*50)

    # 构造 RJ (读状态) 命令: E9 + addr + 01(len) + 52 4A(PDU) + fcs
    pdu = bytes([0x52, 0x4A])  # 'RJ' = read status
    header = bytes([0xE9, addr, len(pdu)])
    fcs = xor_fcs(bytes([addr, len(pdu)]) + pdu)
    frame = header + pdu + bytes([fcs])

    print(f"  发送 RJ(读状态): {frame.hex(' ')}")
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    time.sleep(0.2)

    response = ser.read(20)
    if len(response) > 0:
        print(f"  收到 {len(response)} 字节: {response.hex(' ')}")
        # OEM 协议响应格式: E9 + addr + len + pdu + fcs
        if response[0] == 0xE9:
            resp_addr = response[1]
            resp_len = response[2]
            resp_pdu = response[3:3 + resp_len]
            print(f"    OEM协议响应! 地址={resp_addr}, PDU={resp_pdu.hex(' ')}")
            return True
        elif response[0] == 0xE9:
            print(f"    收到 E9 帧头但格式异常")
        else:
            print(f"    收到数据但不是 OEM 协议格式")
    else:
        print(f"  无响应")

    # 再试 WJ 命令 (设速度但不起动，看泵是否回应)
    print()
    speed = 1000  # 10.00 rpm (单位 0.01rpm 或 0.1rpm)
    pdu_wj = bytes([0x57, 0x4A,              # WJ
                    (speed >> 8) & 0xFF, speed & 0xFF,  # 速度
                    0x00,                     # Bit0=停止, Bit1=正常速度
                    0x01])                    # 方向=CW
    header_wj = bytes([0xE9, addr, len(pdu_wj)])
    fcs_wj = xor_fcs(bytes([addr, len(pdu_wj)]) + pdu_wj)
    frame_wj = header_wj + pdu_wj + bytes([fcs_wj])

    print(f"  发送 WJ(设参不启动): {frame_wj.hex(' ')}")
    ser.reset_input_buffer()
    ser.write(frame_wj)
    ser.flush()
    time.sleep(0.2)

    response2 = ser.read(20)
    if len(response2) > 0:
        print(f"  收到 {len(response2)} 字节: {response2.hex(' ')}")
        if response2[0] == 0xE9:
            print(f"    OEM协议有响应!")
            return True
    else:
        print(f"  无响应")

    return False

# ============================================================
# 变波特率测试
# ============================================================

def test_baudrate_scan():
    """扫描不同波特率，尝试找到泵的实际波特率"""
    ports = Find_COM.list_serial_ports()
    if not ports:
        print("未找到 Serial Port 串口")
        return None, None

    port = ports[0]
    baudrates = [9600, 19200, 38400, 115200, 2400, 4800]
    parities = [
        (serial.PARITY_EVEN, "偶校验(E)"),
        (serial.PARITY_NONE, "无校验(N)"),
        (serial.PARITY_ODD, "奇校验(O)"),
    ]

    print("\n" + "="*50)
    print("【测试0】波特率 + 校验位扫描")
    print("="*50)

    for parity, parity_name in parities:
        for baud in baudrates:
            try:
                test_ser = serial.Serial(
                    port, baudrate=baud,
                    bytesize=serial.EIGHTBITS,
                    parity=parity,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.2
                )
            except Exception as e:
                print(f"  {port} @ {baud} {parity_name}: 打开失败 ({e})")
                continue

            # 发送 Modbus 读寄存器命令
            frame = bytes([1, 0x03, 0x00, 0x20, 0x00, 0x01])
            frame += crc16_modbus(frame)
            test_ser.reset_input_buffer()
            test_ser.write(frame)
            test_ser.flush()
            time.sleep(0.15)
            resp = test_ser.read(20)
            test_ser.close()

            if len(resp) > 0:
                print(f"  {port} @ {baud} {parity_name}: ★ 收到 {len(resp)} 字节 → {resp.hex(' ')}")
                return baud, parity
            else:
                print(f"  {port} @ {baud} {parity_name}: 无响应")

    print("\n  所有波特率/校验位组合均无响应")
    return None, None


# ============================================================
# 主诊断流程
# ============================================================

def main():
    print("兰格 GX 蠕动泵通讯诊断工具")
    print("=" * 50)

    # 1. 查找端口
    ports = Find_COM.list_serial_ports()
    if not ports:
        print("\n错误: 未找到 Serial Port 串口设备!")
        print("请检查:")
        print("  1. Serial Port USB 线是否已插入电脑")
        print("  2. 驱动是否已安装")
        print("  3. 设备管理器中 COM 端口是否显示正常")
        return

    port = ports[0]
    print(f"\n找到泵串口: {port}")

    # 2. 扫描波特率
    baud, parity = test_baudrate_scan()
    if baud is None:
        # 用默认值继续
        baud = 9600
        parity = serial.PARITY_EVEN
        print("\n使用默认参数继续测试: 9600 8E1")

    # 3. 用找到(或默认)的参数打开串口
    print(f"\n最终测试参数: 波特率={baud}, 校验={parity}")
    ser = serial.Serial(
        port, baudrate=baud,
        bytesize=serial.EIGHTBITS,
        parity=parity,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.3
    )

    # 4. 测试 Modbus RTU
    modbus_ok = test_modbus(ser, slave=1)

    # 如果 Modbus 不响应，尝试其他从机地址
    if not modbus_ok:
        for addr in [2, 3, 31]:
            print(f"\n  尝试从机地址 {addr}...")
            if test_modbus(ser, slave=addr):
                modbus_ok = True
                print(f"\n  ★ 发现泵地址为 {addr}! 请将代码中 MODBUS_SLAVE_ADDR 改为 {addr}")
                break

    # 5. 测试 OEM 协议
    oem_ok = test_oem_protocol(ser, addr=1)

    # 6. 总结
    print("\n" + "="*50)
    print("【诊断结果】")
    print("="*50)

    if modbus_ok:
        print("✓ Modbus RTU 协议: 通讯正常")
    else:
        print("✗ Modbus RTU 协议: 无响应")

    if oem_ok:
        print("✓ 兰格 OEM 协议: 通讯正常")
    else:
        print("✗ 兰格 OEM 协议: 无响应")

    if not modbus_ok and not oem_ok:
        print("\n两种协议均无法通讯，请逐一排查:")
        print("  1. 泵是否已通电? 面板屏幕亮吗?")
        print("  2. RS485 线是否接对? A→A(+), B→B(-)")
        print("     (可以试一下把 A/B 线对调)")
        print("  3. 泵面板的通讯参数是否与电脑一致?")
        print("     (进入泵菜单查看地址、波特率、校验位)")
        print("  4. 泵是否处于'通讯控制'模式? (非面板手动模式)")
        print("  5. Serial Port 转接头本身是否正常?")
        print("     (用 sscom32 串口调试工具发数据，看 TX 灯是否闪)")
        print()
        print("如果以上都确认无误但仍不通，可能是泵型号的协议")
        print("与我查到的文档有差异，建议从兰格官网下载对应型号的")
        print("完整通讯协议手册: https://www.longerpump.com.cn")

    ser.close()
    print("\n诊断完成。")


if __name__ == "__main__":
    main()
