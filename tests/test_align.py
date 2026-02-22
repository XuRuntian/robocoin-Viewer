import sys
import os
import numpy as np
import cv2
import json
from PIL import Image
from sklearn.cluster import KMeans  # 引入 K-Means 进行物理状态聚类
from sklearn.preprocessing import StandardScaler

# 确保能导入 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.adapters.lerobot_adapter import LeRobotAdapter

class KinematicScreener:
    """运动学筛选器：基于底层数据寻找高能活跃区和代表性关键帧"""
    def __init__(self, fps=30):
        self.fps = fps

    def compute_energy(self, qpos_data):
        # 计算一阶导数（速度），求平方和代表动能
        velocity = np.diff(qpos_data, axis=0, prepend=qpos_data[0:1])
        energy = np.sum(velocity ** 2, axis=1)
        return energy

    def get_active_window(self, qpos_data, window_size=15, sensitivity=2.0):
        """寻找去除发呆时间的真正活跃区间"""
        raw_energy = self.compute_energy(qpos_data)
        window = np.ones(window_size) / window_size
        smooth_energy = np.convolve(raw_energy, window, mode='same')
        
        # 动态计算底噪 (取最安静的 5% 的帧的平均能量)
        noise_floor = np.mean(np.sort(smooth_energy)[:int(max(1, len(smooth_energy)*0.05))]) 
        noise_floor = max(noise_floor, 1e-6) 
        threshold = noise_floor * sensitivity
        
        active_indices = np.where(smooth_energy > threshold)[0]
        if len(active_indices) == 0:
            return 0, len(qpos_data) - 1

        start_frame = active_indices[0]
        end_frame = active_indices[-1]
        
        # 安全余量 Padding (前后各多给 0.5 秒，给 VLM 提供起手式上下文)
        padding = int(0.5 * self.fps)
        start_frame = max(0, start_frame - padding)
        end_frame = min(len(qpos_data) - 1, end_frame + padding)
        print(f"🔍 运动学筛选: 活跃区间 [{start_frame} -> {end_frame}] (能量阈值: {threshold:.4f})")
        return start_frame, end_frame

    def select_key_frames_kmeans(self, qpos_data, active_start, active_end, num_frames=9):
        """升级版：融合时间嵌入与速度权重的 K-Means 物理状态聚类"""
        active_qpos = qpos_data[active_start:active_end+1]
        
        if len(active_qpos) < num_frames:
            return np.linspace(active_start, active_end, num_frames, dtype=int).tolist()
            
        # 1. 基础特征：位置特征归一化
        scaler_pos = StandardScaler()
        qpos_scaled = scaler_pos.fit_transform(active_qpos)
        # 2. 动势特征：速度特征归一化
        velocities = np.diff(active_qpos, axis=0, prepend=active_qpos[0:1])
        scaler_vel = StandardScaler()
        vel_scaled = scaler_vel.fit_transform(velocities)
        
        # 3. 时序特征：Temporal Embedding
        time_steps = np.arange(len(active_qpos)).reshape(-1, 1)
        scaler_time = StandardScaler()
        time_scaled = scaler_time.fit_transform(time_steps)
        
        # 4. 🧠 核心魔法：多模态特征加权拼接
        W_pos = 1.0   
        W_vel = 2.0   
        W_time = 1.5  
        
        features = np.hstack([
            qpos_scaled * W_pos, 
            vel_scaled * W_vel, 
            time_scaled * W_time
        ])
        
        # 5. 执行聚类
        kmeans = KMeans(n_clusters=num_frames, random_state=42, n_init=10)
        kmeans.fit(features)
        
        # 6. 寻找最贴近聚类中心的真实帧
        key_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features - center, axis=1)
            closest_idx = np.argmin(distances) + active_start
            key_indices.append(closest_idx)
            
        key_indices = sorted(list(set(key_indices)))
        
        while len(key_indices) < num_frames:
            fallback = np.linspace(active_start, active_end, num_frames, dtype=int).tolist()
            key_indices = sorted(list(set(key_indices + fallback)))[:num_frames]
            
        return key_indices

# ==========================================
# 👇 泛用型微观物理对齐引擎
# ==========================================
def find_exact_transition_frame(qpos_window, global_start_idx, gripper_dim_indices, gripper_threshold=0.02):
    """
    底层小脑：寻找精确的动作切换帧 (支持灵巧手与单/双维夹爪)
    基于所有末端维度的复合速度 (L2 Norm) 来感知动作突变。
    """
    if len(qpos_window) < 3 or not gripper_dim_indices:
        return global_start_idx + len(qpos_window) // 2

    # 过滤掉越界的维度，防止报错
    valid_gripper_dims = [d for d in gripper_dim_indices if -qpos_window.shape[1] <= d < qpos_window.shape[1]]
    if not valid_gripper_dims:
        return global_start_idx + len(qpos_window) // 2

    # 策略 A: 寻找末端复合动作突变点 (支持灵巧手协同运动)
    # 1. 提取所有末端维度的状态
    gripper_data = qpos_window[:, valid_gripper_dims]
    
    # 2. 计算末端整体的复合速度 (L2范数)
    gripper_diff = np.diff(gripper_data, axis=0)
    composite_velocity = np.linalg.norm(gripper_diff, axis=1)
    
    if len(composite_velocity) > 0:
        max_change_idx = np.argmax(composite_velocity)
        max_change_val = composite_velocity[max_change_idx]

        if max_change_val > gripper_threshold:
            return global_start_idx + max_change_idx

    # 策略 B: 如果末端没动，寻找手臂动能极小值点 (动作停顿/过渡点)
    all_dims = set(range(qpos_window.shape[1]))
    arm_dims = list(all_dims - set(valid_gripper_dims))
    
    if not arm_dims: # 极端情况：全是末端维度
        return global_start_idx + len(qpos_window) // 2

    arm_qpos = qpos_window[:, arm_dims]
    arm_velocity = np.diff(arm_qpos, axis=0)
    energy = np.sum(arm_velocity ** 2, axis=1)
    
    if len(energy) > 0:
        bottleneck_idx = np.argmin(energy)
        return global_start_idx + bottleneck_idx
    
    return global_start_idx + len(qpos_window) // 2

def align_and_segment(vlm_json, indices_rel, qpos_data, dataset_start_offset, gripper_dim_indices, gripper_threshold):
    """将 VLM 的宏观 JSON 对齐到底层数据"""
    final_annotations = []
    print("\n🚀 开始进行物理-语义缝合...")
    
    for i, task in enumerate(vlm_json):
        img_start_idx = task["start_image"] - 1
        img_end_idx = task["end_image"] - 1
        
        # 使用 K-Means 输出的相对索引
        rough_frame_start = indices_rel[img_start_idx]
        rough_frame_end = indices_rel[img_end_idx]
        
        # 切割对应的底层物理数据
        window_qpos = qpos_data[rough_frame_start : rough_frame_end + 1]
        
        # 精搜切割点，传入指定的末端维度和阈值
        exact_end = find_exact_transition_frame(
            window_qpos, 
            global_start_idx=rough_frame_start,
            gripper_dim_indices=gripper_dim_indices,
            gripper_threshold=gripper_threshold
        )
        
        if i == 0:
            exact_start = rough_frame_start
        else:
            exact_start = final_annotations[-1]["exact_end_frame_relative"]
            
        global_start = dataset_start_offset + exact_start
        global_end = dataset_start_offset + exact_end
        
        final_annotations.append({
            "subtask_id": task["subtask_id"],
            "instruction": task["instruction"],
            "exact_start_frame_relative": int(exact_start),
            "exact_end_frame_relative": int(exact_end),
            "global_start_frame": int(global_start),
            "global_end_frame": int(global_end)
        })
        
        print(f"✅ 子任务 {task['subtask_id']}: {task['instruction']}")
        print(f"   VLM 定界: 图 {task['start_image']} -> {task['end_image']} (预估: {dataset_start_offset+rough_frame_start} -> {dataset_start_offset+rough_frame_end})")
        print(f"   🔪 精准切分: 第 {global_start} 帧 -> 第 {global_end} 帧\n")

    return final_annotations

# ==========================================
# 👇 图像排版工具
# ==========================================
def draw_text_cv2(img_array, text, position=(20, 50), font_scale=1.5, thickness=3, text_color=(255, 255, 255)):
    img = img_array.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return img

def create_optimal_composite_frame(frame_images, step_num):
    global_cam = None
    for key in ['cam_high_rgb', 'cam_third_view', 'front']:
        if key in frame_images and frame_images[key] is not None:
            global_cam = frame_images[key]
            break
    if global_cam is None:
        global_cam = next(iter(frame_images.values()))

    local_cam = None
    for key in ['cam_right_wrist_rgb', 'cam_left_wrist_rgb', 'wrist']:
        if key in frame_images and frame_images[key] is not None:
            local_cam = frame_images[key]
            break
    if local_cam is None:
        local_cam = global_cam  

    target_size = (640, 480)
    global_cam = cv2.resize(global_cam, target_size)
    local_cam = cv2.resize(local_cam, target_size)

    global_cam = draw_text_cv2(global_cam, f"[{step_num}] Global", position=(20, 60), text_color=(255, 50, 50))
    local_cam = draw_text_cv2(local_cam, f"Wrist", position=(20, 60), text_color=(255, 255, 255))

    return np.vstack((global_cam, local_cam))

# ==========================================
# 👇 主函数：Robo-ETL 核心管线
# ==========================================
def main():
    # --- 🤖 硬件参数配置区 ---
    # 根据你的机器人末端类型在此手动指定维度和阈值
    ROBOT_CONFIG = {
        # 示例 A (单臂/双臂简单夹爪): [-1] 或 [-1, -2]
        # 示例 B (灵巧手): [14, 15, 16, 17, 18, 19, 20]
        "gripper_dim_indices": list(range(12, 36)),
        
        # 动作判定阈值: 灵巧手由于单关节位移小，建议降低到 0.02; 简单夹爪可保持 0.05
        "gripper_threshold": 0.05        
    }
    # --------------------------

    dataset_path = "/home/shwu/xrt/test_data/AIRBOT_MMK2_mobile_phone_storage/"
    reader = LeRobotAdapter()
    if not reader.load(dataset_path):
        return
        
    total_length = reader.get_length()
    print(f"📊 数据集总帧数: {total_length}")
    
    EP_START = 0
    EP_END = min(194, total_length - 1) 
    
    print("🔍 正在提取当前 Episode 的底层物理数据...")
    qpos_list = []
    
    for idx in range(EP_START, EP_END + 1):
        frame = reader.get_frame(idx)
        state = getattr(frame, 'state', {})
        val = state.get("qpos")
        if val is None:
            val = state.get("action") 
        if val is not None:
            qpos_list.append(val)
        else:
            # 兜底：如果都没有，给个 0 向量防止崩溃
            qpos_list.append(np.zeros(6))
        
    qpos_data = np.array(qpos_list) # shape: (EP_Length, Dims)
    
    screener = KinematicScreener(fps=30)
    active_start_rel, active_end_rel = screener.get_active_window(qpos_data, window_size=15, sensitivity=2.0)
    
    print("🧠 正在进行 K-Means 物理状态聚类...")
    indices_rel = screener.select_key_frames_kmeans(qpos_data, active_start_rel, active_end_rel, num_frames=9)
    indices = [EP_START + idx for idx in indices_rel]
    
    print(f"✂️ 原始粗暴截断: [0 -> {EP_END}]")
    print(f"🎯 物理引擎动态锁定: 活跃区间 [{EP_START + active_start_rel} -> {EP_START + active_end_rel}]")
    print(f"📸 最终提纯的 9 个关键帧索引: {indices}")

    combo_images = []
    for i, idx in enumerate(indices):
        frame = reader.get_frame(idx)
        if frame and hasattr(frame, 'images') and frame.images:
            combo_arr = create_optimal_composite_frame(frame.images, step_num=i+1)
            combo_images.append(combo_arr)
            print(f"🖼️ 成功渲染动作节点: 步骤 {i+1} (原始帧 {idx})")
            
    reader.close()
    
    if len(combo_images) == 9:
        row1 = np.hstack(combo_images[0:3])
        row2 = np.hstack(combo_images[3:6])
        row3 = np.hstack(combo_images[6:9])
        master_grid = np.vstack((row1, row2, row3))
        
        max_dim = max(master_grid.shape[0], master_grid.shape[1])
        if max_dim > 3000:
            scale = 3000 / max_dim
            master_grid = cv2.resize(master_grid, (0,0), fx=scale, fy=scale)
            
        output_file = "task_test_optimal_grid.jpg"
        Image.fromarray(master_grid).save(output_file, quality=90)
        print(f"\n🎉 完美！K-Means物理对齐排版已生成: {output_file}")
    
    # ==========================================
    # 🌟🌟 核心集成：填入你从网页端获取的 JSON
    # ==========================================
    print("\n[Mock] 正在接收 VLM 网页端返回的语义指令...")
    mock_vlm_json = [
        {
            "subtask_id": 1,
            "instruction": "Left hand approaches and grasps the phone on the left",
            "start_image": 1,
            "end_image": 2
        },
        {
            "subtask_id": 2,
            "instruction": "Left hand lifts the phone, moves it to the center, and places it on the black base",
            "start_image": 2,
            "end_image": 5
        },
        {
            "subtask_id": 3,
            "instruction": "Right hand approaches and grasps the handset of the phone.",
            "start_image": 5,
            "end_image": 6
        },
        {
            "subtask_id": 4,
            "instruction": "Right hand lifts the handset off the base.",
            "start_image": 6,
            "end_image": 7
        },
        {
            "subtask_id": 5,
            "instruction": "Right hand moves the handset to the right and places it on the table.",
            "start_image": 7,
            "end_image": 9
        }
    ]

    # 执行物理缝合，传入在主函数顶部配置好的硬件参数
    final_labels = align_and_segment(
        mock_vlm_json, 
        indices_rel, 
        qpos_data, 
        dataset_start_offset=EP_START,
        gripper_dim_indices=ROBOT_CONFIG["gripper_dim_indices"],
        gripper_threshold=ROBOT_CONFIG["gripper_threshold"]
    )

    print("\n===========================================")
    print("🏆 恭喜！Robo-ETL 最终输出数据集可用的标准格式：")
    
    clean_output = [
        {
            "subtask_id": r["subtask_id"], 
            "instruction": r["instruction"], 
            "start_frame": r["global_start_frame"], 
            "end_frame": r["global_end_frame"]
        } for r in final_labels
    ]
    print(json.dumps(clean_output, indent=2, ensure_ascii=False))
    print("===========================================\n")

if __name__ == "__main__":
    main()