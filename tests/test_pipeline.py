import sys
import os
import time
import datetime
import rerun as rr
import rerun.blueprint as rrb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.inspector import DatasetInspector
from src.core.factory import ReaderFactory
from src.core.reviewer import DatasetReviewer
from src.ui.rerun_visualizer import RerunVisualizer

def save_report(root_dir, bad_datasets):
    """将异常数据列表保存到文件"""
    if not bad_datasets: return None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(root_dir, f"cleaning_report_{timestamp}.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Robocoin Data Cleaning Report\n# Date: {datetime.datetime.now()}\n")
            for path in bad_datasets: f.write(f"{path}\n")
        print(f"\n📄 [报告已生成]: {report_path}")
        return report_path
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")
        return None

def setup_comparison_layout(sample_names, cameras):
    """动态生成并排对比的蓝图"""
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
            name=f"Sample {idx+1}: {name}"
        ))

    blueprint = rrb.Blueprint(
        rrb.Horizontal(*columns),
        collapse_panels=True
    )
    rr.send_blueprint(blueprint)

def run_pipeline():
    # 1. 目标目录 (请根据实际情况修改)
    TARGET_DIR = "/home/user/test_data/hdf5"
    
    viz = RerunVisualizer("RoboCoin_Pipeline_Final")

    # === STEP 1: 格式检查 ===
    print("\n[STEP 1] 格式检查...")
    inspector = DatasetInspector(TARGET_DIR)
    inspector.scan()
    if not inspector.check_consistency(): return

    valid_paths = inspector.get_all_valid_paths()
    print(f"✅ 待审核数据: {len(valid_paths)} 条")

    # === STEP 2: 交互审核 ===
    reviewer = DatasetReviewer(viz)
    bad_datasets = reviewer.start_review(valid_paths)
    final_paths = [p for p in valid_paths if p not in bad_datasets]
    
    print("\n" + "="*50)
    print(f"🎉 审核完成！保留 {len(final_paths)} / {len(valid_paths)} 条有效数据")
    if bad_datasets:
        save_report(TARGET_DIR, bad_datasets)
    else:
        print("✨ 完美！没有发现异常数据。")
    print("="*50)

    if not final_paths: return

    # === STEP 3: 最终预览 (全量帧并行播放) ===
    user_input = input("\n▶️ 是否并排播放 3 条样本进行最终确认？(y/n): ")
    if user_input.lower() == 'y':
        # 选取首、中、尾 3 个样本
        indices = [0]
        if len(final_paths) > 1: indices.append(len(final_paths)-1)
        if len(final_paths) > 2: indices.insert(1, len(final_paths)//2)
        
        sample_paths = [final_paths[i] for i in indices]
        sample_names = [os.path.basename(p) for p in sample_paths]
        
        # 1. 获取传感器列表 (用于布局)
        temp_reader = ReaderFactory.get_reader(sample_paths[0])
        temp_reader.load(sample_paths[0])
        cameras = temp_reader.get_all_sensors()
        temp_reader.close()

        # 2. 发送“并排对比”布局蓝图
        print("\n📺 正在配置并行视图...")
        setup_comparison_layout(sample_names, cameras)
        
        # 3. 清理旧数据
        rr.log("preview", rr.Clear(recursive=True))
        rr.log("world", rr.Clear(recursive=True))

        # 4. 预加载所有 reader 并计算最大长度
        readers = []
        max_length = 0 # 记录最长的视频长度
        
        print("📥 正在加载数据源...")
        for path in sample_paths:
            r = ReaderFactory.get_reader(path)
            r.load(path)
            readers.append(r)
            # 更新最大长度
            curr_len = r.get_length()
            if curr_len > max_length:
                max_length = curr_len
        
        print(f"⏳ 正在缓冲全量视频流 (共 {max_length} 帧)... 请稍候")
        
        # 5. 循环每一帧 (直到最长的视频结束)
        start_time = time.time()
        for i in range(max_length):
            rr.set_time_sequence("frame_idx", i) # 所有样本共享时间轴
            
            for s_idx, reader in enumerate(readers):
                # 如果这个视频比较短，已经播完了，就跳过（画面会保持在最后一帧）
                if i >= reader.get_length(): 
                    continue
                
                frame = reader.get_frame(i)
                
                # A. Log 图像
                for cam, img in frame.images.items():
                    rr.log(f"preview/sample_{s_idx}/{cam}", rr.Image(img))
                
                # B. Log 标题信息
                if i == 0:
                    rr.log(f"preview/sample_{s_idx}/info", rr.TextDocument(f"### {os.path.basename(sample_paths[s_idx])}"))

            # 打印进度条
            if i % 10 == 0 or i == max_length - 1:
                progress = (i + 1) / max_length * 100
                sys.stdout.write(f"\r🚀 缓冲进度: {progress:.1f}% ({i+1}/{max_length})")
                sys.stdout.flush()

        # 关闭资源
        for r in readers: r.close()
        
        duration = time.time() - start_time
        print(f"\n\n✅ 缓冲完成！耗时 {duration:.2f}秒")
        print("👉 Rerun 正在同步播放 3 个样本的完整内容。")

    print("\n✅ 流程结束。")

if __name__ == "__main__":
    run_pipeline()