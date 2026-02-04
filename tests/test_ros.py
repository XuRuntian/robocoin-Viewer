# tests/test_ros.py
import time
from src.core.factory import ReaderFactory
from src.ui.rerun_visualizer import RerunVisualizer

def test_pipeline():
    # !!! 替换为你手头的 mcap 或 bag 文件路径 !!!
    FILE_PATH = "/home/user/test_data/mcap/RB250715046_20251127103139929_RAW/RB250715046_20251127103139929_RAW.mcap" 
    
    print(f"🚀 启动测试: {FILE_PATH}")
    
    # 1. 工厂模式自动获取 Adapter
    try:
        reader = ReaderFactory.get_reader(FILE_PATH)
    except Exception as e:
        print(f"❌ 工厂错误: {e}")
        return

    viz = RerunVisualizer(app_name="RoboCoin_ROS_Test")
    
    if not reader.load(FILE_PATH):
        print("❌ 加载失败")
        return

    sensor_list = reader.get_all_sensors()
    print(f"📷 发现传感器: {sensor_list}")
    viz.setup_layout(sensor_list)

    total = reader.get_length()
    print(f"📊 总帧数 (以主相机为准): {total}")

    # 推送前 200 帧测试
    for i in range(total):
        frame = reader.get_frame(i)
        viz.log_frame(frame, i)
        if i % 10 == 0:
            print(f"Processed {i}")

    # 保持运行
    while True:
        time.sleep(1)

if __name__ == "__main__":
    test_pipeline()