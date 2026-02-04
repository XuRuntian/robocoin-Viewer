# src/app.py
import time
from src.core.factory import ReaderFactory
from src.ui.rerun_visualizer import RerunVisualizer

def run_viewer(file_path: str):
    # 1. 获取 Reader (解耦：App 不知道它是 HDF5 还是 MCAP)
    reader = ReaderFactory.get_reader(file_path)
    if not reader.load(file_path):
        return

    # 2. 初始化 Visualizer (预留了 UI 布局)
    viz = RerunVisualizer()

    # 3. 数据流式推送
    print(f"正在同步数据到 Rerun...")
    for i in range(reader.get_length()):
        frame = reader.get_frame(i)
        viz.log_frame(frame, i)
    
    print("同步完成！请在 Rerun 窗口查看。")
    # 保持进程不退出，否则 Rerun 窗口可能会关闭
    while True:
        time.sleep(1)

if __name__ == "__main__":
    # 这里以后可以改为从 UI 界面获取路径
    test_file = "/home/user/test_data/hdf5/episode_1.hdf5"
    run_viewer(test_file)