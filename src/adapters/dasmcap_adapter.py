# src/adapters/ego_adapter.py
import os
import json
import av
import cv2
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
from pathlib import Path
from typing import List, Dict, Optional, Any
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from concurrent.futures import ThreadPoolExecutor

from src.core.interface import BaseDatasetReader, FrameData, AdapterConfig
from src.core.registry import AdapterRegistry

class VideoDecoder:
    """参考 das-datakit 的 H264 连续解码器 (开启了内部多线程)"""
    def __init__(self):
        av.logging.set_level(av.logging.ERROR)
        self.decoder_codec = av.CodecContext.create('h264', 'r')
        self.decoder_codec.thread_count = 4  # 允许FFmpeg底层多核解码
        self.has_find_first_kf = False

    def decode(self, compressed_data: bytes) -> np.ndarray:
        if not compressed_data: return None
        start_offset = 4 if compressed_data.startswith(b"\x00\x00\x00\x01") else 3 if compressed_data.startswith(b"\x00\x00\x01") else 0
        nal_unit_type = compressed_data[start_offset] & 0x1F
        
        if nal_unit_type != 7 and not self.has_find_first_kf:
            return None  
            
        self.has_find_first_kf = True
        try:
            packet = av.packet.Packet(compressed_data)
            for frame in self.decoder_codec.decode(packet):
                # 直接转为 RGB24 格式的 numpy 数组，跳过后续所有色彩空间转换
                return frame.to_ndarray(format="rgb24")
        except Exception:
            return None
        return None

@AdapterRegistry.register("DASMCAP")
class DASMCAPAdapter(BaseDatasetReader):
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.root_path = None
        self.episode_files = []
        self.current_episode_idx = 0
        
        # 1. 基础配置（使用 or {} 确保即使配置里是 None，也会变成空字典）
        self.camera_map = getattr(self.config, 'image_keys_map', {}) or {}
        self.arm_groups = getattr(self.config, 'arm_groups', {}) or {}
        self.base_map = getattr(self.config, 'state_keys_map', {}) or {}
        
        # 额外的开关配置
        extra_opts = getattr(self.config, 'extra_options', {}) or {}
        self.enable_undistort = extra_opts.get("enable_undistort", False)
                
        # 2. 数据缓存初始化
        self.image_keys = []           
        self.timestamps = []           
        self.images_cache = {}         
        self.camera_info_cache = {}    
        self.raw_state_data: Dict[str, List] = {} 
        self.interpolators = {}
        
        # [优化项]: 初始化线程池用于并行解码视频
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 8)

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        self.episode_files = []
        if self.root_path.is_file() and self.root_path.suffix.lower() == '.mcap':
            self.episode_files.append(self.root_path)
        elif self.root_path.is_dir():
            self.episode_files.extend(sorted(self.root_path.rglob("*.mcap")))
        if not self.episode_files:
            print("❌ [DASMCAPAdapter] 未找到任何 .mcap 文件。")
            return False
        self.set_episode(0)
        return True

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episode_files): return
        self.close()  
        self.current_episode_idx = episode_idx
        target_file = self.episode_files[episode_idx]
        print(f"🔄 [DASMCAPAdapter] 解析数据流: {target_file.name}")
        
        all_arm_topics = []
        for g in self.arm_groups.values():
            if not g: continue # 防止 group 本身是 None
            for key in ['pose_topic', 'gripper_topic', 'joint_topic']:
                val = g.get(key)
                if val: all_arm_topics.append(val)
        
        base_topics = list(self.base_map.keys())
        target_topics = set(list(self.camera_map.keys()) + all_arm_topics + base_topics)

        # [优化项]: 临时缓存每路相机的二进制压缩数据
        raw_video_packets: Dict[str, List[tuple]] = {}
        
        with open(target_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, proto_msg in reader.iter_decoded_messages():
                topic = channel.topic
                if topic not in target_topics: continue

                # 1. 解析相机视频流 ([仅提取数据，不解码])
                if topic in self.camera_map:
                    std_cam_name = self.camera_map[topic]
                    if std_cam_name not in raw_video_packets:
                        raw_video_packets[std_cam_name] = []
                        if std_cam_name not in self.image_keys:
                            self.image_keys.append(std_cam_name)
                    # 存入原始 H264 字节流
                    raw_video_packets[std_cam_name].append((message.publish_time, proto_msg.data))

                # 2. 解析末端位姿 (EEF Pose) -> 完全保留你的原逻辑
                elif any(topic == g.get('pose_topic') for g in self.arm_groups.values()):
                    if topic not in self.raw_state_data: self.raw_state_data[topic] = []
                    pose = np.array([
                        proto_msg.pose.position.x, proto_msg.pose.position.y, proto_msg.pose.position.z,
                        proto_msg.pose.orientation.x, proto_msg.pose.orientation.y, proto_msg.pose.orientation.z, proto_msg.pose.orientation.w
                    ])
                    self.raw_state_data[topic].append((message.publish_time, pose))

                # 3. 解析夹爪或通用状态 (Gripper / Base / Joints) -> 完全保留你的原逻辑
                else:
                    if topic not in self.raw_state_data: self.raw_state_data[topic] = []
                    val = getattr(proto_msg, 'value', getattr(proto_msg, 'data', 0.0))
                    # 自动处理 list 或单值
                    val = np.array(val) if isinstance(val, (list, np.ndarray)) else float(val)
                    self.raw_state_data[topic].append((message.publish_time, val))

        # [优化项]: 文件读取完毕，开始按相机并行解码视频流
        def decode_stream(cam_name, packets):
            decoder = VideoDecoder()
            imgs = []
            tms = []
            for pub_time, data in packets:
                img = decoder.decode(data)
                if img is not None:
                    imgs.append(img)
                    tms.append(pub_time)
            return cam_name, imgs, tms

        futures = [self.executor.submit(decode_stream, name, pkts) for name, pkts in raw_video_packets.items()]
        for future in futures:
            name, imgs, tms = future.result()
            self.images_cache[name] = imgs
            # 仅在第一台相机上建立主时间轴
            if self.image_keys and name == self.image_keys[0]:
                self.timestamps = tms

        self._build_interpolators()
        print(f"✅ [DASMCAPAdapter] Episode 加载完毕，长度: {len(self.timestamps)} 帧")

    def _build_interpolators(self):
        self.interpolators.clear()
        for topic, cache in self.raw_state_data.items():
            if len(cache) < 2: continue
            times, unique_idx = np.unique([c[0] for c in cache], return_index=True)
            data = np.array([c[1] for c in cache])[unique_idx]
            
            # 如果是 7 维数据，判定为 Pose，建立 Slerp
            if data.ndim == 2 and data.shape[1] == 7:
                pos = data[:, :3]
                rots = st.Rotation.from_quat(data[:, 3:7])
                self.interpolators[f"{topic}_pos"] = si.interp1d(times, pos, axis=0, bounds_error=False, fill_value=(pos[0], pos[-1]))
                self.interpolators[f"{topic}_rot"] = st.Slerp(times, rots)
                self.interpolators[f"{topic}_rot_bounds"] = (times[0], times[-1])
            else:
                # 普通线性插值
                self.interpolators[topic] = si.interp1d(times, data, axis=0, bounds_error=False, fill_value="extrapolate")

    def get_frame(self, index: int, specific_cameras: Optional[List[str]] = None) -> FrameData:
        if index < 0 or index >= len(self.timestamps): return None
        target_time = self.timestamps[index]
        
        # 1. 获取图像
        images = {}
        keys_to_fetch = specific_cameras if specific_cameras else self.image_keys
        for cam_name in keys_to_fetch:
            if cam_name in self.images_cache and index < len(self.images_cache[cam_name]):
                # [优化项]: 内存中已经是 RGB Numpy 数组，直接读取即可，不需要任何转换
                images[cam_name] = self.images_cache[cam_name][index]

        # 2. 构建 QPos (Bucket 模式) -> 完全保留你的原逻辑
        qpos_list = []
        
        # --- 遍历所有手臂组 (Arms & Grippers) ---
        for arm_name in sorted(self.arm_groups.keys()):
            group = self.arm_groups[arm_name]
            
            # EEF Pose (7维)
            pt = group.get('pose_topic')
            if pt and f"{pt}_pos" in self.interpolators:
                pos = self.interpolators[f"{pt}_pos"](target_time)
                t_min, t_max = self.interpolators[f"{pt}_rot_bounds"]
                safe_t = np.clip(target_time, t_min, t_max)
                rot = self.interpolators[f"{pt}_rot"](safe_t).as_quat()
                qpos_list.extend(pos.tolist() + rot.tolist())
            
            # Joints (多维)
            jt = group.get('joint_topic')
            if jt and jt in self.interpolators:
                joints = self.interpolators[jt](target_time)
                qpos_list.extend(joints.tolist() if joints.ndim > 0 else [float(joints)])

            # Gripper (1维)
            gt = group.get('gripper_topic')
            if gt and gt in self.interpolators:
                grip = self.interpolators[gt](target_time)
                qpos_list.append(float(grip*10)) # 夹爪变为0-1之间而非0-0.1

        # --- 遍历底座 (Base) ---
        for topic in self.base_map.keys():
            if topic in self.interpolators:
                base_data = self.interpolators[topic](target_time)
                qpos_list.extend(base_data.tolist() if base_data.ndim > 0 else [float(base_data)])

        return FrameData(timestamp=target_time/1e9, images=images, state={'qpos': np.array(qpos_list)})
    
    def get_all_sensors(self) -> List[str]:
        """返回当前加载的 episode 中所有的传感器(相机)名称"""
        return self.image_keys

    def get_current_episode_path(self) -> str:
        """返回当前正在读取的 episode 的文件路径"""
        if not self.episode_files or self.current_episode_idx >= len(self.episode_files):
            return ""
        return str(self.episode_files[self.current_episode_idx])

    def get_length(self) -> int:
        """返回当前 episode 的总帧数"""
        return len(self.timestamps)

    def get_total_episodes(self) -> int:
        """返回当前数据集包含的总 episode 数量"""
        return len(self.episode_files)
    
    def close(self):
        self.images_cache.clear()
        self.camera_info_cache.clear()
        self.image_keys.clear()
        
        self.timestamps.clear()
        
        if hasattr(self, 'raw_state_data'):
            self.raw_state_data.clear()
        self.interpolators.clear()