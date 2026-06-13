#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: Find_COM.py
Author: Zinc Zou
Email: zinczou@163.com
Date: 2024/10/11
Copyright: 慕乐网络科技(大连）有限公司
        www.mools.net
        moolsnet@126.com
Description: 串口自动检测工具
"""
import serial.tools.list_ports


def list_serial_ports():
    """查找所有 CH340 串口设备"""
    ports = serial.tools.list_ports.comports()
    serial_ports_list = []
    for port in ports:
        if 'CH340' in port.description or 'CH340' in port.device:
            serial_ports_list.append(port.device)
            print("Found CH340 Port:", port.device)
    if serial_ports_list:
        return serial_ports_list
    else:
        return []


def list_USB_ports():
    """查找所有 USB 串口设备"""
    ports = serial.tools.list_ports.comports()
    USB_ports_list = []
    for port in ports:
        if '串行' in port.description or '串行' in port.device:
            USB_ports_list.append(port.device)
            print("Found USB ports:", port.device)
    if USB_ports_list:
        return USB_ports_list
    else:
        return []


if __name__ == "__main__":
    ports = list(serial.tools.list_ports.comports())
    if len(ports) == 0:
        print('No port available')
    else:
        for port in ports:
            print(port)

    serial_ports = list_serial_ports()
    if serial_ports:
        port = serial_ports[0]
        print(f"使用 CH340 Port: {port}")
        pump_ser = serial.Serial(
            port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5
        )
        print(f"已连接 {port}，波特率 9600，8E1 (Modbus RTU)")
    else:
        print("未找到 CH340 串口设备，跳过串口连接测试")

    USB_ports = list_USB_ports()
    if USB_ports:
        port_USB = USB_ports[0]
        baudrate = 115200
        USB_ser = serial.Serial(port_USB, baudrate)
        print(f"已连接 USB 端口: {port_USB}，波特率 {baudrate}")
    else:
        print("未找到 USB 串口设备，跳过 USB 连接测试")
