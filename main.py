# main.py
import sys
import os
import argparse
import time
from pathlib import Path

# 确保能找到 src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.core.inspector import DatasetInspector
from src.core.factory import ReaderFactory
from src.core.reviewer import DatasetReviewer
from src.core.config_generator import ConfigGenerator
from src.core.organizer import DatasetOrganizer
from src.ui.rerun_visualizer import RerunVisualizer
import rerun as rr
import rerun.blueprint as rrb

def setup_comparison_layout(sample_names, cameras):
    """
    配置并排对比视图的蓝图
    """
    columns = []
    for idx, name in enumerate(sample_names):
        cam_views = []
        for cam in cameras:
            cam_views.append(rrb.Spatial2DView(
                origin=f"preview/sample_{idx}/{cam}",
                name=f"{cam}"
            ))
        columns.append(rrb.Vertical(
            rrb.TextDocumentView(origin=f"preview/sample_{idx}/info", name=f"{name}"),
            *cam_views,
            name=f"Sample {idx+1}"
        ))
    
    blueprint = rrb.Blueprint(rrb.Horizontal(*columns), collapse_panels=True)
    rr.send_blueprint(blueprint)

def save_report(root_dir, bad_datasets):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root_dir, f"cleaning_report_{timestamp}.txt")
    try:
        with open(path, "w") as f:
            f.write("# Bad Datasets Report\n")
            for p in bad_datasets: f.write(f"{p}\n")
        print(f"📄 异常报告已保存: {path}")
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")

def main():
    # 1. 命令行参数定义
    parser = argparse.ArgumentParser(description="RoboCoin Viewer - 具身智能数据集清洗与预览工具")
    parser.add_argument("path", type=str, help="数据集根目录路径 (例如: ./data/hdf5)")
    parser.add_argument("--skip-review", action="store_true", help="跳过【交互式审核】步骤，直接预览")
    parser.add_argument("--no-preview", action="store_true", help="流程结束后不播放预览视频")
    
    args = parser.parse_args()
    TARGET_DIR = args.path
    
    if not os.path.exists(TARGET_DIR):
        print(f"❌ 错误: 路径不存在 -> {TARGET_DIR}")
        return

    # 初始化 Rerun 应用
    viz = RerunVisualizer("RoboCoin_Main")

    # === STEP 1: 格式检查 (Inspector) ===
    print("\n🔍 [1/4] 正在扫描目录格式...")
    inspector = DatasetInspector(TARGET_DIR)
    inspector.scan()
    
    # 如果一致性检查失败，直接退出
    if not inspector.check_consistency():
        return 

    # 初始化 Organizer
    organizer = DatasetOrganizer(TARGET_DIR)
    
    # 检查是否需要自动整理
    grouped_datasets = inspector.grouped_datasets
    if len(grouped_datasets) > 1:
        print(f"🔄 检测到多种类型数据集: {list(grouped_datasets.keys())}")
        new_grouped_paths = organizer.sort_by_type(grouped_datasets, TARGET_DIR)
        # 更新有效路径为整理后的新路径
        valid_paths = []
        for paths in new_grouped_paths.values():
            valid_paths.extend(paths)
        print(f"✅ 数据集已整理到类型分组文件夹中")
    else:
        valid_paths = inspector.get_all_valid_paths()
        
    print(f"✅ 有效数据集: {len(valid_paths)} 条")

    # === STEP 2: 交互审核 (Reviewer) ===
    final_paths = valid_paths
    
    if not args.skip_review:
        # 启动审核器 (键盘控制: N/P/Space/Esc)
        reviewer = DatasetReviewer(viz)
        bad_datasets = reviewer.start_review(valid_paths)
        
        if bad_datasets:
            # 使用 Organizer 进行物理隔离
            quarantine_dir = organizer.quarantine_bad_data(bad_datasets, TARGET_DIR)
            print(f"🔒 异常数据已隔离到: {quarantine_dir}")
            # 剔除坏数据，保留好数据进入下一步
            final_paths = [p for p in valid_paths if p not in bad_datasets]
            print(f"🧹 剔除异常数据后剩余: {len(final_paths)} 条")
        else:
            print("✨ 完美！未发现异常数据。")
    else:
        print("⏩ 已跳过交互审核步骤。")

    if not final_paths:
        print("❌ 没有有效数据可供后续处理。")
        return

    # === STEP 3: 并行预览 (Preview) ===
    if not args.no_preview:
        # 自动选取 3 个样本 (首、中、尾)
        indices = [0]
        if len(final_paths) > 1: indices.append(len(final_paths)-1)
        if len(final_paths) > 2: indices.insert(1, len(final_paths)//2)
        sample_paths = [final_paths[i] for i in indices]
        
        # 准备元数据 (获取相机列表)
        temp_reader = ReaderFactory.get_reader(sample_paths[0])
        temp_reader.load(sample_paths[0])
        cameras = temp_reader.get_all_sensors()
        
        print("\n📺 [3/4] 正在准备并行预览...")
        setup_comparison_layout([os.path.basename(p) for p in sample_paths], cameras)
        temp_reader.close()
        
        # 清理旧画面
        rr.log("preview", rr.Clear(recursive=True))
        rr.log("world", rr.Clear(recursive=True))
        
        # 预加载所有 reader 并计算最大长度
        readers = []
        max_len = 0
        print("📥 正在缓冲视频流...")
        for p in sample_paths:
            r = ReaderFactory.get_reader(p)
            r.load(p)
            readers.append(r)
            if r.get_length() > max_len: max_len = r.get_length()
        
        print(f"▶️ 正在同步播放 {len(readers)} 个样本 (Max Frames: {max_len})...")
        
        # 播放循环
        for i in range(max_len):
            rr.set_time_sequence("frame_idx", i)
            
            for s_idx, r in enumerate(readers):
                if i >= r.get_length(): continue
                
                frame = r.get_frame(i)
                # Log 图像
                for cam, img in frame.images.items():
                    rr.log(f"preview/sample_{s_idx}/{cam}", rr.Image(img))
                
                # Log 标题 (仅第0帧)
                if i == 0:
                    rr.log(f"preview/sample_{s_idx}/info", rr.TextDocument(f"### {os.path.basename(sample_paths[s_idx])}"))

            # 简单的进度打印
            if i % 30 == 0: print(".", end="", flush=True)

        for r in readers: r.close()
        print("\n✅ 预览播放完成。")

    # === STEP 4: 生成配置 (ConfigGenerator) ===
    print("\n📝 [4/4] 检查配置生成接口...")
    # 使用第一个样本来调用接口
    sample_reader = ReaderFactory.get_reader(final_paths[0])
    sample_reader.load(final_paths[0])
    
    # 这里调用我们刚写的“空接口”
    ConfigGenerator.analyze_and_save(sample_reader, TARGET_DIR)
    sample_reader.close()

    print("\n🎉 全部流程结束！")

if __name__ == "__main__":
    main()
