# src/core/ai_screener.py
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

class AIScreener:
    """
    基于 CLIP 模型的 AI 数据筛查器
    功能：通过提取每条 Episode 的中间帧特征向量，检测混入的其他任务或废片
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def detect_outliers(self, reader, outlier_ratio=0.1, similarity_threshold=0.85):
        """
        传入已经 load 好的 reader，返回判定为异常的 episode_idx 列表
        """
        features = {}
        total_eps = reader.get_total_episodes()
        
        # 1. 提取每条轨迹的特征
        for ep_idx in range(total_eps):
            try:
                reader.set_episode(ep_idx)
                length = reader.get_length()
                if length < 10:
                    continue
                    
                # 采样 3 帧取平均特征
                indices = [int(length * 0.1), int(length * 0.5), int(length * 0.9)]
                frame_vectors = []
                
                for idx in indices:
                    frame = reader.get_frame(idx)
                    if frame is None or not frame.images: continue
                    
                    # 随便取一个存在的相机视角
                    image_arr = next(iter(frame.images.values()))
                    image = Image.fromarray(image_arr).convert('RGB')
                    
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        v = self.model.get_image_features(**inputs).cpu().numpy().flatten()
                        frame_vectors.append(v)
                
                if frame_vectors:
                    features[ep_idx] = np.mean(frame_vectors, axis=0)
            except Exception as e:
                print(f"⚠️ Episode {ep_idx} 特征提取失败: {e}")
                
        # 2. 计算离群值
        if len(features) < 3:
            return [] # 样本太少，不做聚类剔除
            
        feature_matrix = np.stack(list(features.values()))
        ep_indices = np.array(list(features.keys()))
        
        centroid = np.mean(feature_matrix, axis=0)
        similarities = cosine_similarity(feature_matrix, [centroid]).flatten()
        
        sorted_idx = np.argsort(similarities)
        outlier_count = max(1, int(len(features) * outlier_ratio))
        
        suspect_indices = sorted_idx[:outlier_count]
        low_similarity_mask = similarities[suspect_indices] < similarity_threshold
        
        # 返回被判定为异常的 Episode ID 列表
        suspect_eps = ep_indices[suspect_indices][low_similarity_mask]
        return suspect_eps.tolist()