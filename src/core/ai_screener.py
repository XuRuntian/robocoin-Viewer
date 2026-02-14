import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from src.core.factory import ReaderFactory
from PIL import Image
import os

class AIScreener:
    """
    基于CLIP模型的AI数据筛查器
    功能：通过提取数据集中间帧的特征向量，使用余弦相似度检测离群数据
    """
    
    def __init__(self):
        """初始化模型、处理器和设备配置"""
        # 自动检测计算设备
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # 加载CLIP模型和处理器（使用vit-base-patch32架构）
        self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def _get_image_from_dataset(self, path: str):
        """
        从数据集中提取中间帧图像
        参数:
            path: 数据集文件路径
        返回:
            PIL.Image.Image 对象或 None（失败时）
        """
        try:
            # 获取数据集读取器
            reader = ReaderFactory.get_reader(path)
            if not reader:
                raise ValueError(f"无法识别文件格式: {path}")
                
            # 加载数据集并获取中间帧
            reader.load(path)
            mid_idx = reader.get_length() // 2
            
            # 提取图像数据
            frame = reader.get_frame(mid_idx)
            
            # 兼容性防御：处理空帧或解码失败的情况
            if frame is None or not hasattr(frame, 'images') or not frame.images:
                print(f"⚠️ 无法从 [{os.path.basename(path)}] 提取图像 (可能由于服务器缺少视频解码器)")
                reader.close()
                return None
                
            images = frame.images
            
            # 优先选择指定视角的图像
            image = None
            for key in ['cam_high', 'front', 'image', 'camera']:
                if key in images:
                    image = images[key]
                    break
            
            # 如果没找到指定视角，取第一个可用图像
            if image is None and images:
                image = next(iter(images.values()))
                
            reader.close()
            
            # 二次防御
            if image is None:
                return None
            
            # 确保返回PIL图像对象
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            return image.convert('RGB')
                
        except Exception as e:
            print(f"❌ 图像读取失败 [{path}]: {str(e)}")
            return None

    def extract_embeddings(self, dataset_paths: list):
        """
        提取数据集中间帧的CLIP特征向量
        参数:
            dataset_paths: 数据集文件路径列表
        返回:
            字典 {路径: 特征向量}
        """
        embeddings = {}
        total = len(dataset_paths)
        
        for i, path in enumerate(dataset_paths):
            # 获取图像
            image = self._get_image_from_dataset(path)
            if image is None:
                continue
                
            try:
                # 特征提取
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                
                # 转换为numpy数组并展平
                embeddings[path] = outputs.cpu().numpy().flatten()
                
                # 进度显示
                print(f"🧠 [AI 提取中] {i+1}/{total}: {os.path.basename(path)}")
                
            except Exception as e:
                print(f"❌ 特征提取失败 [{path}]: {str(e)}")
                continue
                
        return embeddings

    def detect_outliers(self, dataset_paths: list, outlier_ratio=0.05, similarity_threshold=0.85):
        """
        离群数据检测核心方法
        参数:
            dataset_paths: 数据集文件路径列表
            outlier_ratio: 离群比例阈值（默认5%）
            similarity_threshold: 相似度绝对阈值（默认0.85）
        返回:
            可疑路径列表
        """
        # 提取特征向量
        features = self.extract_embeddings(dataset_paths)
        if len(features) < 3:
            print("⚠️ 样本数量不足，无法进行离群检测")
            return []
            
        # 特征矩阵构建
        feature_matrix = np.stack(list(features.values()))
        paths = np.array(list(features.keys()))
        
        # 计算中心向量
        centroid = np.mean(feature_matrix, axis=0)
        
        # 计算余弦相似度
        similarities = cosine_similarity(feature_matrix, [centroid]).flatten()
        
        # 按相似度排序（从小到大）
        sorted_indices = np.argsort(similarities)
        
        # 离群判定逻辑：
        # 1. 首先按相似度排序取最低的outlier_ratio比例
        # 2. 再过滤出相似度低于threshold的样本
        outlier_count = max(1, int(len(features) * outlier_ratio))
        suspect_indices = sorted_indices[:outlier_count]
        low_similarity_mask = similarities[suspect_indices] < similarity_threshold
        
        # 最终可疑样本
        suspects = paths[suspect_indices][low_similarity_mask]
        
        # 打印检测结果
        print(f"\n🔍 离群检测完成:")
        print(f"📊 总样本数: {len(features)}")
        print(f"📉 相似度阈值: {similarity_threshold}")
        print(f"🎯 离群比例: {outlier_ratio*100}% ({outlier_count}个)")
        print(f"🚨 检测到可疑样本: {len(suspects)} 个")
        print("\n".join([f" - {os.path.basename(p)} (相似度: {similarities[i]:.3f})" 
                        for i, p in zip(suspect_indices[low_similarity_mask], suspects)]))
        
        return list(suspects)

if __name__ == "__main__":
    # 测试示例
    screener = AIScreener()
    test_paths = [
        "data/valid/episode_0.hdf5",
        "data/valid/episode_1.hdf5",
        "data/valid/episode_2.hdf5",
        "data/valid/episode_3.hdf5",
        "data/valid/episode_4.hdf5",
        "data/valid/episode_5.hdf5",
        "data/valid/episode_6.hdf5",
        "data/valid/episode_7.hdf5",
        "data/valid/episode_8.hdf5",
        "data/valid/episode_9.hdf5"
    ]
    outliers = screener.detect_outliers(test_paths)
    print("\n✅ 最终可疑数据路径:")
    for path in outliers:
        print(path)
