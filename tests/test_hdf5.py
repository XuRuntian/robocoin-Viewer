# tests/test_hdf5_to_rerun.py
import time
import sys
import os

# 确保能找到 src 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adapters.hdf5_adapter import HDF5Adapter
from src.ui.rerun_visualizer import RerunVisualizer

def test_pipeline():
    # 1. 配置路径 (替换为你自己的真实路径)
    FILE_PATH = "/home/user/test_data/hdf5/episode_1.hdf5"
    
    print(f"🚀 启动测试: {FILE_PATH}")
    
    # 2. 初始化组件
    reader = HDF5Adapter()
    viz = RerunVisualizer(app_name="RoboCoin_HDF5_Test")
    
    # 3. 加载数据
    if not reader.load(FILE_PATH):
        print("❌ HDF5 加载失败，请检查文件路径和格式")
        return
    # === 新增逻辑开始 ===
    # 4. 获取元数据并配置布局
    # 从 Reader 获取所有传感器名字 (我们在 Interface 里定义过 get_all_sensors)
    sensor_list = reader.get_all_sensors()
    print(f"📷 发现传感器: {sensor_list}")
    
    # 告诉 Visualizer 根据这些传感器生成界面
    viz.setup_layout(sensor_list)
    # === 新增逻辑结束 ===
    total_frames = reader.get_length()
    print(f"📊 检测到 {total_frames} 帧数据，准备推送至 Rerun...")

    # 4. 循环推送数据
    # 为了测试效率，我们可以只推送前 500 帧，或者全部推送
    start_time = time.time()
    for i in range(total_frames):
        try:
            # 获取一帧
            frame = reader.get_frame(i)
            
            # 推送到 Rerun
            viz.log_frame(frame, frame_idx=i)
            
            # 每 100 帧打印一次进度
            if i % 100 == 0:
                print(f"已处理: {i}/{total_frames}")
                
        except Exception as e:
            print(f"❌ 处理第 {i} 帧时出错: {e}")
            break

    end_time = time.time()
    print(f"✅ 完成！耗时: {end_time - start_time:.2f}秒")
    print("👉 请在 Rerun 窗口中操作时间轴进行预览。")

    # 保持进程，否则 Rerun 窗口会随脚本结束而关闭
    print("按 Ctrl+C 退出测试...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        reader.close()
        print("\n测试结束。")

if __name__ == "__main__":
    test_pipeline()