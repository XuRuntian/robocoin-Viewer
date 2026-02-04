# tests/test_unitree.py
import time
import sys
import os

# 确保能找到 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.factory import ReaderFactory
from src.ui.rerun_visualizer import RerunVisualizer

def test_unitree_pipeline():
    # 1. 设置路径 (指向包含 data.json 的那个文件夹)
    # 根据你之前的 ls 结果，路径应该是这个：
    DATA_PATH = "/home/user/test_data/lerobot/episode_0000"
    
    print(f"🚀 启动 Unitree 数据测试: {DATA_PATH}")
    
    # 2. 使用工厂获取 Reader
    # 工厂会检测文件夹下有没有 data.json 且 author=unitree
    try:
        reader = ReaderFactory.get_reader(DATA_PATH)
        print(f"✅ 成功匹配 Adapter: {type(reader).__name__}")
    except Exception as e:
        print(f"❌ 工厂匹配失败: {e}")
        return

    # 3. 加载数据
    if not reader.load(DATA_PATH):
        print("❌ 数据加载失败，请检查 data.json 是否完整")
        return

    # 4. 打印元数据验证
    total_frames = reader.get_length()
    sensors = reader.get_all_sensors()
    print(f"📊 数据概览:")
    print(f"   - 总帧数: {total_frames}")
    print(f"   - 传感器 (相机): {sensors}")
    
    # 5. 启动可视化
    viz = RerunVisualizer(app_name="RoboCoin_Unitree_Test")
    viz.setup_layout(sensors)
    
    # 6. 读取并推送前 200 帧 (或全部)
    print("▶️ 开始推送数据到 Rerun...")
    
    # 简单检查第 0 帧的数据结构
    first_frame = reader.get_frame(0)
    print(f"   [Debug 第0帧状态数据 Keys]: {list(first_frame.state.keys())}")
    if 'left_ee_tactile' in first_frame.state:
        print(f"   [Debug 触觉数据形状]: {first_frame.state['left_ee_tactile'].shape}")

    # 循环播放
    for i in range(min(total_frames, 300)): # 先测 300 帧
        frame = reader.get_frame(i)
        viz.log_frame(frame, i)
        
        # 模拟 30fps 的播放速度，不然跑太快了
        # time.sleep(0.03) 
        
        if i % 50 == 0:
            print(f"   已处理: {i}/{total_frames}")

    print("✅ 测试完成！请在 Rerun 窗口查看。")
    print("   - 检查是否有 2 个相机画面 (color_0, color_1)")
    print("   - 检查下方是否有 qpos 波形图")
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("退出。")

if __name__ == "__main__":
    test_unitree_pipeline()