import sys
import os
import numpy as np
import cv2
from PIL import Image

# 确保能导入 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.adapters.lerobot_adapter import LeRobotAdapter

def draw_text_cv2(img_array, text, position=(20, 50), font_scale=1.5, thickness=3, text_color=(255, 255, 255)):
    """使用 OpenCV 绘制带黑色背景框的高对比度文本"""
    img = img_array.copy()
    # 计算文字背景框大小
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    # 画黑色半透明/实心背景框
    cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
    # 画文字
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return img

def create_optimal_composite_frame(frame_images, step_num):
    """提取 Global 和 Wrist 两个核心视角，上下拼接"""
    # 1. 寻找全局视角
    global_cam = None
    for key in ['cam_high_rgb', 'cam_third_view', 'front']:
        if key in frame_images and frame_images[key] is not None:
            global_cam = frame_images[key]
            break
    if global_cam is None:
        global_cam = next(iter(frame_images.values()))

    # 2. 寻找腕部/局部视角
    local_cam = None
    for key in ['cam_right_wrist_rgb', 'cam_left_wrist_rgb', 'wrist']:
        if key in frame_images and frame_images[key] is not None:
            local_cam = frame_images[key]
            break
    if local_cam is None:
        local_cam = global_cam  # 没找到腕部就用全局凑数

    # 3. 强制统一尺寸，防止上下拼接报错 (假设统一到 640x480)
    target_size = (640, 480)
    global_cam = cv2.resize(global_cam, target_size)
    local_cam = cv2.resize(local_cam, target_size)

    # 4. 注入巨型强对比度标签 (RGB 颜色空间：红是 255, 50, 50)
    global_cam = draw_text_cv2(global_cam, f"[{step_num}] Global", position=(20, 60), text_color=(255, 50, 50))
    local_cam = draw_text_cv2(local_cam, f"Wrist", position=(20, 60), text_color=(255, 255, 255))

    # 5. 上下拼接成一个长方形帧
    return np.vstack((global_cam, local_cam))

def main():
    dataset_path = "/home/shwu/xrt/test_data/AIRBOT_MMK2_mobile_phone_storage/"
    reader = LeRobotAdapter()
    if not reader.load(dataset_path):
        return
        
    total_length = reader.get_length()
    print(f"📊 数据集总帧数: {total_length}")
    
    # 【核心修复】锁定单一 Episode 范围
    # 6秒的视频，通常在 180 ~ 300 帧之间。我们这里取前 250 帧作为 Episode 0 的绝对安全范围。
    EP_START = 0
    EP_END = 250  # 如果你发现最后几帧动作还没做完，可以把这个值调大到 300
    
    # 重新计算分布区间
    fractions = [0.0, 0.15, 0.35, 0.45, 0.50, 0.55, 0.65, 0.85, 0.99]
    indices = [EP_START + int((EP_END - EP_START) * f) for f in fractions]
    # 确保不越界
    indices = [min(idx, total_length - 1) for idx in indices]
    
    combo_images = []
    for i, idx in enumerate(indices):
        frame = reader.get_frame(idx)
        if frame and hasattr(frame, 'images') and frame.images:
            combo_arr = create_optimal_composite_frame(frame.images, step_num=i+1)
            combo_images.append(combo_arr)
            print(f"📸 成功提取动作焦点: 步骤 {i+1} (原始帧 {idx})")
            
    reader.close()
    
    # 生成 3x3 黄金比例母图
    if len(combo_images) == 9:
        row1 = np.hstack(combo_images[0:3])
        row2 = np.hstack(combo_images[3:6])
        row3 = np.hstack(combo_images[6:9])
        master_grid = np.vstack((row1, row2, row3))
        
        # 尺寸保护：防止图片超出 VLM 处理上限
        max_dim = max(master_grid.shape[0], master_grid.shape[1])
        if max_dim > 3000:
            scale = 3000 / max_dim
            master_grid = cv2.resize(master_grid, (0,0), fx=scale, fy=scale)
            
        output_file = "task_test_optimal_grid.jpg"
        Image.fromarray(master_grid).save(output_file, quality=90)
        print(f"\n🎉 完美！最优视觉排版已生成: {output_file}")
        
if __name__ == "__main__":
    main()