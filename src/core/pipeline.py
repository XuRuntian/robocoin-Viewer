# src/core/pipeline.py
import numpy as np
import os
from src.core.factory import ReaderFactory
from src.core.kinematics import KinematicScreener, align_and_segment
from src.core.image_utils import GridImageGenerator  
from src.core.vlm_caller import call_qwen_vl_api     

class RoboETLPipeline:
    def __init__(self, dataset_path, robot_config):
        self.dataset_path = dataset_path
        self.robot_config = robot_config
        self.reader = ReaderFactory.get_reader(dataset_path)
        if not self.reader.load(dataset_path):
            raise ValueError("数据集加载失败")

    def extract_qpos(self, ep_length):
        qpos_list = []
        for idx in range(ep_length):
            frame = self.reader.get_frame(idx)
            state = getattr(frame, 'state', {})
            val = state.get("qpos") if state.get("qpos") is not None else state.get("action")
            qpos_list.append(val if val is not None else np.zeros(6))
        return np.array(qpos_list)

    def process_episode(self, episode_idx, task_desc, progress_callback=None, is_suspect=False):
        """处理单条轨迹的完整流水线"""
        self.reader.set_episode(episode_idx)
        ep_length = self.reader.get_length()
        
        # 👇 【新增核心逻辑】如果是疑似废片，直接拦截！省钱、防幻觉、防报错！
        if is_suspect:
            if progress_callback: progress_callback("⚠️ AI 视觉体检异常，已跳过大模型标注，标记为废片。")
            return [{
                "subtask_id": 1, 
                "instruction": "⚠️ [异常废片] 动作或画面偏离主题，建议直接剔除！", 
                "start_frame": 0, 
                "end_frame": max(0, ep_length - 1)
            }]

        # --- 以下为正常的端到端处理逻辑 ---
        if progress_callback: progress_callback("🔄 正在提取底层运动学数据...")
        qpos_data = self.extract_qpos(ep_length)
        
        if progress_callback: progress_callback("🧠 正在进行 K-Means 物理特征聚类...")
        screener = KinematicScreener(fps=30)
        active_start, active_end = screener.get_active_window(qpos_data)
        indices_rel = screener.select_key_frames_kmeans(qpos_data, active_start, active_end)
        absolute_indices = [idx for idx in indices_rel]

        if progress_callback: progress_callback("🖼️ 正在渲染并导出 3x3 关键帧九宫格...")
        temp_grid_path = f"temp_grid_ep{episode_idx}.jpg"
        success = GridImageGenerator.generate_3x3_grid(self.reader, absolute_indices, temp_grid_path)
        if not success:
            raise RuntimeError(f"Episode {episode_idx}: 九宫格生成失败。")

        if progress_callback: progress_callback("🌐 正在请求 Qwen-VL 进行宏观语义拆解...")
        try:
            vlm_json = call_qwen_vl_api(image_path=temp_grid_path, global_task_desc=task_desc)
        finally:
            if os.path.exists(temp_grid_path): os.remove(temp_grid_path)
        
        if progress_callback: progress_callback("⚙️ 正在执行微观物理状态缝合...")
        final_labels = align_and_segment(
            vlm_json=vlm_json, indices_rel=indices_rel, qpos_data=qpos_data,
            dataset_start_offset=0, gripper_dim_indices=self.robot_config["gripper_dim_indices"],
            gripper_threshold=self.robot_config["gripper_threshold"]
        )
        
        return [{"subtask_id": r["subtask_id"], "instruction": r["instruction"], "start_frame": r["global_start_frame"], "end_frame": r["global_end_frame"]} for r in final_labels]

    def close(self):
        self.reader.close()