# src/core/config_generator.py
import os
import uuid
import yaml

class ConfigGenerator:
    @staticmethod
    def generate_yaml_string(data: dict) -> str:
        """根据数据字典生成符合规范的 YAML 字符串"""
        # 确保包含 UUID (把它插到 dataset_name 后面，保证 YAML 顺序美观)
        if 'dataset_uuid' not in data or not data['dataset_uuid']:
            # 用一个小技巧重排字典，让 UUID 紧跟 dataset_name
            new_data = {}
            for k, v in data.items():
                new_data[k] = v
                if k == 'dataset_name':
                    new_data['dataset_uuid'] = str(uuid.uuid4())
            data = new_data
            
        # 自动将字典转换为 YAML 格式
        # sort_keys=False 保证 YAML 输出的顺序和我们在界面上填写的顺序一致
        return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @staticmethod
    def analyze_and_save(data: dict, save_dir: str, filename="dataset_config.yaml"):
        """
        保存标注好的配置到指定目录
        """
        yaml_content = ConfigGenerator.generate_yaml_string(data)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"✅ [ConfigGenerator] 标注文件已保存至: {filepath}")
        return filepath