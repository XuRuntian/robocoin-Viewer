# tests/test_pipeline.py
import sys
import os
import time
import datetime # <--- 新增时间库

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.inspector import DatasetInspector
from src.core.factory import ReaderFactory
from src.core.reviewer import DatasetReviewer
from src.ui.rerun_visualizer import RerunVisualizer
import rerun as rr

def save_report(root_dir, bad_datasets):
    """
    将异常数据列表保存到文件
    """
    if not bad_datasets:
        return None

    # 生成带时间戳的文件名，防止覆盖
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"cleaning_report_{timestamp}.txt"
    report_path = os.path.join(root_dir, report_filename)

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Robocoin Data Cleaning Report\n")
            f.write(f"# Date: {datetime.datetime.now()}\n")
            f.write(f"# Total Bad Datasets: {len(bad_datasets)}\n")
            f.write("-" * 50 + "\n")
            for path in bad_datasets:
                f.write(f"{path}\n")
        
        print(f"\n📄 [报告已生成]: {report_path}")
        return report_path
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")
        return None

def run_pipeline():
    # 1. 目标目录 (请修改为你真实的测试路径)
    TARGET_DIR = "/home/user/test_data/hdf5"
    
    # 初始化 Rerun (只需一次)
    viz = RerunVisualizer("RoboCoin_Pipeline_Final")

    # === STEP 1: 格式一致性检查 (Gatekeeper) ===
    print("\n[STEP 1] 格式检查...")
    inspector = DatasetInspector(TARGET_DIR)
    inspector.scan()
    
    if not inspector.check_consistency():
        print("\n⛔ 流程终止：请先清理数据集中的异常文件。")
        return

    # 获取所有通过初筛的路径
    valid_paths = inspector.get_all_valid_paths()
    print(f"✅ 待审核数据: {len(valid_paths)} 条")

    # === STEP 2: 交互式内容审核 (Reviewer) ===
    # 这里会阻塞，直到用户按 'q' 或审核完成
    reviewer = DatasetReviewer(viz)
    bad_datasets = reviewer.start_review(valid_paths)

    # 剔除坏数据
    final_paths = [p for p in valid_paths if p not in bad_datasets]
    
    print("\n" + "="*50)
    print(f"🎉 审核完成！保留 {len(final_paths)} / {len(valid_paths)} 条有效数据")
    
    # === 新增功能: 保存异常记录到文件 ===
    if bad_datasets:
        print(f"🗑️ 检测到 {len(bad_datasets)} 条异常数据")
        report_file = save_report(TARGET_DIR, bad_datasets)
        
        if report_file:
            print(f"💡 提示: 你可以使用以下命令批量删除这些数据:")
            print(f"   xargs rm -rf < {os.path.basename(report_file)}")
    else:
        print("✨ 完美！没有发现异常数据。")
    print("="*50)

    if not final_paths:
        print("❌ 所有数据都被标记为 Bad，流程结束。")
        return

    # === STEP 3: 最终预览 (Preview) ===
    user_input = input("\n▶️ 是否播放几条样本数据进行最终确认？(y/n): ")
    if user_input.lower() == 'y':
        # 选取 3 个样本 (首、中、尾)
        indices = [0]
        if len(final_paths) > 1: indices.append(len(final_paths)-1)
        if len(final_paths) > 2: indices.insert(1, len(final_paths)//2)
        
        sample_paths = [final_paths[i] for i in indices]
        
        # 重置回标准布局
        sample_reader = ReaderFactory.get_reader(sample_paths[0])
        sample_reader.load(sample_paths[0])
        viz.setup_layout(sample_reader.get_all_sensors()) 
        sample_reader.close()
        
        print("\n正在缓冲视频流...")
        for idx, path in enumerate(sample_paths):
            reader = ReaderFactory.get_reader(path)
            reader.load(path)
            
            ep_name = os.path.basename(path)
            print(f"播放: {ep_name}")
            
            # 播放 150 帧
            for i in range(min(150, reader.get_length())):
                frame = reader.get_frame(i)
                viz.log_frame(frame, idx * 1000 + i)
            
            reader.close()

    print("\n✅ 流程结束。")

if __name__ == "__main__":
    run_pipeline()