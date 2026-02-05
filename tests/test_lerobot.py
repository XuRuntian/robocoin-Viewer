# tests/test_lerobot.py
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.factory import ReaderFactory
from src.ui.rerun_visualizer import RerunVisualizer

def test():
    # 替换为你下载的 LeRobot 数据集路径
    # 结构应该是: /path/to/dataset/data/train-xxx.parquet
    DATA_PATH = "/home/user/test_data/整理线缆与USB插入_Organizing Cables and USB Insertion_486_61854" 
    
    print(f"🚀 测试 LeRobot Adapter: {DATA_PATH}")
    try:
        reader = ReaderFactory.get_reader(DATA_PATH)
        print(f"✅ 成功创建适配器: {type(reader).__name__}")
    except Exception as e:
        print(f"❌ 工厂匹配失败: {e}")
        return

    if not reader.load(DATA_PATH):
        print("❌ 加载失败")
        return

    viz = RerunVisualizer("LeRobot_Test")
    viz.setup_layout(reader.get_all_sensors())
    
    length = reader.get_length()
    print(f"📊 总帧数: {length}")

    # 播放
    for i in range(min(length, 300)):
        frame = reader.get_frame(i)
        viz.log_frame(frame, i)
        if i % 50 == 0: print(f"Frame {i}")
    
    while True: time.sleep(1)

if __name__ == "__main__":
    test()