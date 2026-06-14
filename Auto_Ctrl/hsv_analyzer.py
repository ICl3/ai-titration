import cv2
import numpy as np


class HSVColorAnalyzer:
    """基于 HSV 色彩空间的颜色分析器，作为深度学习模型的备选方案。"""

    def __init__(self, config: dict):
        target = config.get('hsv_target', {})
        # JS 版 H 范围 0-359 → OpenCV H 范围 0-179，折半
        self.h_min = target.get('hMin', 0) / 2
        self.h_max = target.get('hMax', 30) / 2
        self.s_min = target.get('sMin', 0)
        self.s_max = target.get('sMax', 255)
        self.v_min = target.get('vMin', 0)
        self.v_max = target.get('vMax', 255)
        self.roi_size = config.get('roi_size', 200)
        self.endpoint_color = config.get('endpoint_color', 'orange')

    def analyze_image(self, image_path: str):
        """分析图片，返回 (颜色名称, 置信度)。"""
        img = cv2.imread(image_path)
        if img is None:
            return self.endpoint_color, 0.0

        h, w = img.shape[:2]
        size = min(self.roi_size, w, h)
        x1 = max(0, (w - size) // 2)
        y1 = max(0, (h - size) // 2)
        roi = img[y1:y1 + size, x1:x1 + size]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # OpenCV HSV: H[0-179], S[0-255], V[0-255]

        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # H 通道匹配（处理环绕，如红色 hMin > hMax）
        if self.h_min > self.h_max:
            h_match = (h_channel >= self.h_min) | (h_channel <= self.h_max)
        else:
            h_match = (h_channel >= self.h_min) & (h_channel <= self.h_max)

        s_match = (s_channel >= self.s_min) & (s_channel <= self.s_max)
        v_match = (v_channel >= self.v_min) & (v_channel <= self.v_max)

        target_pixels = np.count_nonzero(h_match & s_match & v_match)
        total_pixels = size * size
        confidence = target_pixels / total_pixels if total_pixels > 0 else 0.0

        return self.endpoint_color, float(confidence)
