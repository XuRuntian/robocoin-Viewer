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

    def generate_global_template(self, task_desc, sample_size=3, progress_callback=None):
        """
        [彻底重构] 物理级多合一：仅调用 1 次 API
        """
        total_eps = self.reader.get_total_episodes()
        sample_indices = np.linspace(0, total_eps - 1, sample_size, dtype=int)
        
        # 收集采样配置
        sample_configs = []
        if progress_callback:
            progress_callback(f"🧪 正在采集 {len(sample_indices)} 条参考轨迹的关键帧...")

        for ep_idx in sample_indices:
            self.reader.set_episode(ep_idx)
            qpos = self.extract_qpos(self.reader.get_length())
            screener = KinematicScreener()
            start, end = screener.get_active_window(qpos)
            indices = screener.select_key_frames_kmeans(qpos, start, end)
            sample_configs.append((ep_idx, indices))

        # 1. 物理拼接成一张超级大图
        mega_path = "mega_template_grid.jpg"
        GridImageGenerator.generate_mega_grid(self.reader, sample_configs, mega_path)

        # 2. 修改 Prompt，告诉 AI 同时看这三组数据
        mega_prompt = (
            f"{task_desc}\n"
            f"图中垂直排列了 {sample_size} 组相同任务的执行过程。每一组都是 3x3 九宫格。\n"
            "请观察这些不同实例的共性，给出最通用的子任务拆解逻辑（Subtask JSON）。"
        )

        try:
            if progress_callback: progress_callback("🌐 正在发送超级大图，请求全局任务标准 (Only 1 API Call)...")
            # 只有这里调一次 API
            master_template = call_qwen_vl_api(image_path=mega_path, global_task_desc=mega_prompt)
            return master_template
        finally:
            if os.path.exists(mega_path): os.remove(mega_path)

    def process_with_template(self, episode_idx, template, is_suspect=False):
        """
        [核心优化] 利用本地算法将轨迹对齐到全局模板。
        """
        self.reader.set_episode(episode_idx)
        ep_length = self.reader.get_length()
        
        if is_suspect:
            return [{
                "subtask_id": 1, 
                "instruction": "⚠️ [异常废片] 请复核", 
                "start_frame": 0, 
                "end_frame": ep_length - 1
            }]

        qpos_data = self.extract_qpos(ep_length)
        screener = KinematicScreener()
        start, end = screener.get_active_window(qpos_data)
        indices_rel = screener.select_key_frames_kmeans(qpos_data, start, end)
        
        # 1. 拿到包含完整调试信息的原始标签
        raw_labels = align_and_segment(
            vlm_json=template, 
            indices_rel=indices_rel, 
            qpos_data=qpos_data,
            dataset_start_offset=0,
            gripper_dim_indices=self.robot_config["gripper_dim_indices"],
            gripper_threshold=self.robot_config["gripper_threshold"]
        )

        # 2. 👇 【关键修复】在此处过滤掉不希望在 UI 展示的列
        # 只保留前端绘图和展示需要的核心 4 列
        clean_labels = [
            {
                "subtask_id": r["subtask_id"],
                "instruction": r["instruction"],
                "start_frame": r["start_frame"],
                "end_frame": r["end_frame"]
            } 
            for r in raw_labels
        ]
        
        return clean_labels
    
    def close(self):
        self.reader.close()