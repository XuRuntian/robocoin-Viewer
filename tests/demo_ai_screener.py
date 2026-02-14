import sys
import os
import argparse
from pathlib import Path

# 确保能导入 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.ai_screener import AIScreener
from src.core.inspector import DatasetInspector

def main():
    parser = argparse.ArgumentParser(description="AI 离群数据检测演示")
    parser.add_argument("path", type=str, help="数据集根目录 (例如: /home/user/test_data/hdf5)")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"❌ 路径不存在: {args.path}")
        return

    # 1. 先用 Inspector 扫出所有有效数据
    print("\n🔍 第一步：扫描目录寻找数据集...")
    inspector = DatasetInspector(args.path)
    inspector.scan()
    valid_paths = inspector.get_all_valid_paths()
    
    if len(valid_paths) < 3:
        print("⚠️ 数据集数量太少（<3），无法进行有意义的 AI 离群检测。")
        return
        
    print(f"✅ 找到 {len(valid_paths)} 个有效数据集。\n")

    # 2. 启动 AI 筛查
    print("🤖 第二步：启动 AI 视觉特征筛查...")
    print("首次运行可能需要下载 CLIP 模型权重，请耐心等待...")
    screener = AIScreener()
    
    # 提取并计算离群值
    # 这里我们将阈值设得稍微敏感一点，方便在测试集中看到效果
    suspects = screener.detect_outliers(valid_paths, outlier_ratio=0.1, similarity_threshold=0.85)
    
    # 3. 结果汇总
    print("\n" + "="*40)
    print("🏁 AI 筛查工作流演示结束")
    print("="*40)
    if suspects:
        print("建议将以下数据送入 Reviewer 进行【重点人工复核】:")
        for p in suspects:
            print(f" ⚠️ {p}")
    else:
        print("✨ 未发现明显的离群异常数据。")

if __name__ == "__main__":
    main()