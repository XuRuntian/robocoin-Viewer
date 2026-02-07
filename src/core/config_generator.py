# src/core/config_generator.py
from src.core.interface import BaseDatasetReader

class ConfigGenerator:
    @staticmethod
    def analyze_and_save(reader: BaseDatasetReader, save_dir: str, filename="dataset_config.yaml"):
        """
        [接口预留]
        目前仅打印提示，不执行实际生成逻辑。
        未来可在此处实现：分析相机分辨率、FPS、状态维度等，并生成 yaml。
        """
        # 这里什么都不做，只打印一条日志
        print(f"ℹ️ [ConfigGenerator] YAML 生成功能暂未启用 (接口已预留)。")
        return None