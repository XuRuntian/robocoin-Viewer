import os
import json
import cv2
import numpy as np
import scipy.interpolate as si
from pathlib import Path
from typing import List, Dict, Optional
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from google.protobuf.json_format import MessageToDict 

# 👉 [FK新增] 引入 URDF 和四元数计算库
from scipy.spatial.transform import Rotation as R
try:
    from yourdfpy import URDF
except ImportError:
    URDF = None

from src.core.interface import BaseDatasetReader, FrameData, AdapterConfig
from src.core.registry import AdapterRegistry
import logging

logger = logging.getLogger(__name__)

@AdapterRegistry.register("SingorixHybrid")
class SingorixHybridAdapter(BaseDatasetReader):
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        
        self.camera_map = getattr(self.config, 'image_keys_map', {}) or {}
        self.arm_groups = getattr(self.config, 'arm_groups', {}) or {}
        self.base_map = getattr(self.config, 'state_keys_map', {}) or {}
        
        self.episode_files = []
        self.current_episode_idx = 0
        self.video_handles: Dict[str, cv2.VideoCapture] = {}
        
        self.interpolators = {}         
        self.action_interpolators = {}  
        
        self.fps = 30.0
        self.first_mcap_time = 0.0
        self._length = 0

        # 👉 [FK新增] 初始化 URDF 相关变量
        extra_opts = getattr(self.config, 'extra_options', {}) or {}
        self.urdf_path = extra_opts.get('urdf_path', None)
        self.target_links = extra_opts.get('target_links', ["left_arm_link7", "right_arm_link7"])
        self.robot = None
        self.urdf_joints = {"left": [], "right": []}

    def load(self, file_path: str) -> bool:
        root_path = Path(file_path)
        if root_path.is_file(): root_path = root_path.parent
            
        self.episode_files = []
        if (root_path / "data_collection_meta.json").exists():
            self.episode_files.append(root_path)
        else:
            for meta_file in root_path.rglob("data_collection_meta.json"):
                self.episode_files.append(meta_file.parent)
                
        if not self.episode_files:
            logger.error("❌ [SingorixHybrid] 未找到 data_collection_meta.json")
            return False
            
        self.set_episode(0)
        return True

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episode_files): return
        self.close()
        self.current_episode_idx = episode_idx
        self.first_mcap_time = 0.0
        ep_dir = self.episode_files[episode_idx]
        logger.info(f"🔄 [SingorixHybrid] 加载 Episode: {ep_dir.name}")

        # 👉 [FK新增] 顺便加载 URDF 模型
        if self.urdf_path and self.robot is None:
            if URDF is None:
                logger.warning("⚠️ [SingorixHybrid] 未安装 yourdfpy，跳过 URDF/FK 计算")
            elif os.path.exists(self.urdf_path):
                try:
                    self.robot = URDF.load(self.urdf_path)
                    joint_names = self.robot.actuated_joint_names
                    self.urdf_joints["left"] = [j for j in joint_names if "left_arm" in j]
                    self.urdf_joints["right"] = [j for j in joint_names if "right_arm" in j]
                    logger.info(f"✅ [SingorixHybrid] URDF 模型加载成功: {self.urdf_path}")
                except Exception as e:
                    logger.error(f"❌ [SingorixHybrid] URDF 加载失败: {e}")
            else:
                logger.warning(f"⚠️ [SingorixHybrid] 找不到 URDF 文件: {self.urdf_path}")

        # 1. 解析 Metadata 获取 FPS
        meta_file = ep_dir / "data_collection_meta.json"
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            self.fps = meta.get("tag", {}).get("fps", 30.0)

        # 2. 挂载视频句柄
        for mp4_filename, std_cam_name in self.camera_map.items():
            vid_path = ep_dir / mp4_filename
            if vid_path.exists():
                cap = cv2.VideoCapture(str(vid_path))
                self.video_handles[std_cam_name] = cap
                if self._length == 0:
                    self._length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 3. 解析 MCAP 状态数据
        mcap_files = list(ep_dir.glob("*.mcap"))
        if not mcap_files: return
            
        raw_state_data = {}
        raw_action_data = {}
        
        with open(mcap_files[0], "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, proto_msg in reader.iter_decoded_messages():
                t = message.publish_time / 1e9

                if channel.topic == "singorix/wbcs/sensor":
                    if self.first_mcap_time == 0.0:
                        self.first_mcap_time = t
                    
                    # 🚀 优化 2.1: 尝试绕过耗时的 MessageToDict，直接访问 proto 对象 (Fast-Path)
                    joint_map = getattr(proto_msg, "joint_sensor_map", getattr(proto_msg, "jointSensorMap", None))
                    if joint_map is not None:
                        for group_name, info in joint_map.items():
                            if group_name not in raw_state_data: raw_state_data[group_name] = []
                            pos = np.array(list(info.position))  # 直接转换
                            if "gripper" in group_name.lower(): pos = pos / 1000.0 
                            raw_state_data[group_name].append((t, pos))
                    else:
                        # 退回原始逻辑 (Fallback)
                        msg_dict = MessageToDict(proto_msg, always_print_fields_with_no_presence=True)
                        if "jointSensorMap" in msg_dict:
                            for group_name, info in msg_dict["jointSensorMap"].items():
                                if group_name not in raw_state_data: raw_state_data[group_name] = []
                                pos = np.array(info.get("position", []))
                                if "gripper" in group_name.lower(): pos = pos / 1000.0 
                                raw_state_data[group_name].append((t, pos))

                elif channel.topic == "singorix/wbcs/target":
                    # 🚀 优化 2.2: 同样应用 Fast-Path
                    target_map = getattr(proto_msg, "target_group_trajectory_map", getattr(proto_msg, "targetGroupTrajectoryMap", None))
                    if target_map is not None:
                        for group_name, trajectory_info in target_map.items():
                            if group_name not in raw_action_data: raw_action_data[group_name] = []
                            group_cmds = trajectory_info.group_commands if hasattr(trajectory_info, "group_commands") else trajectory_info.groupCommands
                            if group_cmds:
                                joint_cmds = group_cmds[0].joint_commands if hasattr(group_cmds[0], "joint_commands") else group_cmds[0].jointCommands
                                positions = np.array([cmd.position for cmd in joint_cmds])
                                if "gripper" in group_name.lower(): positions = positions / 1000.0
                                raw_action_data[group_name].append((t, positions))
                    else:
                        # 退回原始逻辑 (Fallback)
                        msg_dict = MessageToDict(proto_msg, always_print_fields_with_no_presence=True)
                        if "targetGroupTrajectoryMap" in msg_dict:
                            for group_name, trajectory_info in msg_dict["targetGroupTrajectoryMap"].items():
                                if group_name not in raw_action_data: raw_action_data[group_name] = []
                                group_cmds = trajectory_info.get("groupCommands", [])
                                if group_cmds:
                                    joint_cmds = group_cmds[0].get("jointCommands", [])
                                    positions = np.array([cmd.get("position", 0.0) for cmd in joint_cmds])
                                    if "gripper" in group_name.lower(): positions = positions / 1000.0
                                    raw_action_data[group_name].append((t, positions))

        # 4. 构建插值器 (Qpos)
        self.interpolators.clear()
        for group_name, cache in raw_state_data.items():
            if len(cache) < 2: continue
            times = np.array([c[0] for c in cache])
            data = np.array([c[1] for c in cache])
            self.interpolators[group_name] = si.interp1d(times, data, axis=0, bounds_error=False, fill_value=(data[0], data[-1]))

        # 5. 构建插值器 (Action)
        self.action_interpolators.clear()
        for group_name, cache in raw_action_data.items():
            if len(cache) < 2: continue
            times = np.array([c[0] for c in cache])
            data = np.array([c[1] for c in cache])
            self.action_interpolators[group_name] = si.interp1d(times, data, axis=0, bounds_error=False, fill_value=(data[0], data[-1]))

        logger.info(f"✅ Episode 加载完毕，总帧数: {self._length}, FPS: {self.fps:.2f}")

    def get_frame(self, index: int, specific_cameras: Optional[List[str]] = None) -> FrameData:
        if index < 0 or index >= self._length: return None
        target_time = self.first_mcap_time + (index / self.fps)

        # 1. 读取图像
        images = {}
        keys_to_fetch = specific_cameras if specific_cameras else list(self.camera_map.values())
        for std_cam_name in keys_to_fetch:
            if std_cam_name in self.video_handles:
                cap = self.video_handles[std_cam_name]
                
                # 🚀 优化 1: 避免 OpenCV 昂贵的重新寻址。如果是顺序拿帧（最常见），直接 read 会极速提升。
                current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame_pos != index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    
                ret, frame = cap.read()
                if ret: images[std_cam_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. 构建 QPos 和 Action 状态向量
        qpos_list = []
        action_list = []
        
        # 👉 [FK新增] 准备收集当前帧用于计算 FK 的配置字典
        urdf_cfg = {}
        
        # --- 遍历手臂和夹爪 ---
        for arm_name in sorted(self.arm_groups.keys()):
            group = self.arm_groups[arm_name]
            jt = group.get('joint_topic')
            gt = group.get('gripper_topic')
            
            # Qpos 组装
            if jt and jt in self.interpolators: 
                arm_qpos = self.interpolators[jt](target_time)
                qpos_list.extend(arm_qpos.tolist())
                
                # 👉 [FK新增] 将插值出来的关节角度映射给 URDF
                if self.robot:
                    prefix = "left" if "left" in arm_name.lower() else "right"
                    expected_jts = self.urdf_joints.get(prefix, [])
                    if len(arm_qpos) == len(expected_jts):
                        for i, j_name in enumerate(expected_jts):
                            urdf_cfg[j_name] = arm_qpos[i]
                            
            if gt and gt in self.interpolators: 
                qpos_list.extend(self.interpolators[gt](target_time).tolist())
            
            # Action 组装
            if jt and jt in self.action_interpolators: action_list.extend(self.action_interpolators[jt](target_time).tolist())
            if gt and gt in self.action_interpolators: action_list.extend(self.action_interpolators[gt](target_time).tolist())

        # --- 遍历底座和其他部位 ---
        for config_key, mcap_key in self.base_map.items():
            if mcap_key in self.interpolators:
                base_data = self.interpolators[mcap_key](target_time)
                qpos_list.extend(base_data.tolist() if base_data.ndim > 0 else [float(base_data)])
                
            if mcap_key in self.action_interpolators:
                act_data = self.action_interpolators[mcap_key](target_time)
                action_list.extend(act_data.tolist() if act_data.ndim > 0 else [float(act_data)])

        # 👉 [FK新增] 实时计算 EE Pose
        eepose_list = []
        if self.robot and urdf_cfg:
            self.robot.update_cfg(urdf_cfg)
            
            for link in self.target_links:
                if link in self.robot.scene.graph.nodes:
                    ee_trans = self.robot.scene.graph.get(link)[0]
                    trans = ee_trans[:3, 3]
                    rot_mat = ee_trans[:3, :3]
                    
                    # 旋转矩阵转四元数 [x, y, z, w] -> 优化：一次性转出不再创建多余对象
                    euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)
                    
                    eepose_list.extend(trans) 
                    eepose_list.extend(euler) 

        if eepose_list:
            qpos_list.extend(eepose_list)
            
        state_dict = {
            'qpos': np.array(qpos_list, dtype=np.float32),
            'action': np.array(action_list, dtype=np.float32) if action_list else None
        }
        
        return FrameData(
            timestamp=target_time, 
            images=images, 
            state=state_dict
        )

    def get_all_sensors(self) -> List[str]: return list(self.video_handles.keys())
    def get_length(self) -> int: return self._length
    def get_total_episodes(self) -> int: return len(self.episode_files)
    def get_current_episode_path(self) -> str: return str(self.episode_files[self.current_episode_idx]) if self.episode_files else ""
    
    def close(self):
        for cap in self.video_handles.values(): cap.release()
        self.video_handles.clear()
        self.interpolators.clear()
        if hasattr(self, 'action_interpolators'):
            self.action_interpolators.clear()
        self._length = 0
