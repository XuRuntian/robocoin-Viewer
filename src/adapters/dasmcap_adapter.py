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

from src.core.interface import BaseDatasetReader, FrameData, AdapterConfig
from src.core.registry import AdapterRegistry

class VideoDecoder:
    """参考 das-datakit 的 H264 连续解码器"""
    def __init__(self):
        av.logging.set_level(av.logging.ERROR)
        self.decoder_codec = av.CodecContext.create('h264', 'r')
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
        
        # 1. 基础配置
        self.enable_undistort = False
        self.rule_name = "default"
        self.rules_file_path = "adapter_rules.json"
        
        if self.config and self.config.extra_options:
            self.enable_undistort = self.config.extra_options.get("enable_undistort", False)
            self.rule_name = self.config.extra_options.get("rule_name", "default")
            self.rules_file_path = self.config.extra_options.get("rules_file", "adapter_rules.json")
        
        # 2. 动态映射表 (由 load_rules 填充)
        self.camera_map = {}   # e.g., {"/robot0/sensor/camera0/compressed": "left_wrist_cam"}
        self.pose_map = {}     # e.g., {"/robot0/vio/eef_pose": "left_ee"}
        self.gripper_map = {}  # e.g., {"/robot0/sensor/magnetic_encoder": "left_gripper"}
        self._load_rules()
        
        # 3. 数据缓存
        self.image_keys = []           
        self.timestamps = []           
        self.images_cache = {}         
        self.camera_info_cache = {}    
        
        # 针对多设备的独立缓存
        self.poses_cache: Dict[str, List] = {name: [] for name in set(self.pose_map.values())}
        self.grippers_cache: Dict[str, List] = {name: [] for name in set(self.gripper_map.values())}
        self.interpolators = {}

    def _load_rules(self):
        """从 adapter_rules.json 加载指定设备的 Topic 映射关系"""
        if not os.path.exists(self.rules_file_path):
            print(f"⚠️ [DASMCAPAdapter] 警告: 找不到规则文件 {self.rules_file_path}，将降级为原版匹配逻辑。")
            return
            
        try:
            with open(self.rules_file_path, "r", encoding="utf-8") as f:
                all_rules = json.load(f)
                
            if self.rule_name not in all_rules:
                raise ValueError(f"规则名称 '{self.rule_name}' 不在配置文件中！可用规则: {list(all_rules.keys())}")
                
            rule = all_rules[self.rule_name]
            self.camera_map = rule.get("cameras", {})
            self.pose_map = rule.get("poses", {})
            self.gripper_map = rule.get("grippers", {})
            print(f"🔧 [DASMCAPAdapter] 成功加载映射规则 [{self.rule_name}]")
            
        except Exception as e:
            print(f"❌ [DASMCAPAdapter] 加载规则失败: {e}")

    @staticmethod
    def _generate_ds_map_numerical(width, height, fu, fv, cu, cv, xi, alpha):
        # ... (此处保持你原有的去畸变映射生成逻辑) ...
        u_out, v_out = np.meshgrid(np.arange(width), np.arange(height))
        u_out, v_out = u_out.astype(np.float64), v_out.astype(np.float64)
        u_out_f, v_out_f = u_out.astype(np.float32), v_out.astype(np.float32)
        x, y, z = (u_out_f - cu) / fu, (v_out_f - cv) / fv, np.ones_like(u_out_f)
        d1 = np.sqrt(x*x + y*y + z*z)
        mz = (1.0 - alpha) * d1 + alpha * z
        d2 = np.sqrt(x*x + y*y + mz*mz)
        denominator = np.clip((1.0 - alpha) * d2 + alpha * (xi * d1 + z), 1e-8, None)
        u_in, v_in = fu * x / denominator + cu, fv * y / denominator + cv
        return u_in.astype(np.float32), v_in.astype(np.float32)

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
            
        print(f"✅ [DASMCAPAdapter] 扫描到 {len(self.episode_files)} 个数据包")
        self.set_episode(0)
        return True

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episode_files): return
        
        self.close()  
        self.current_episode_idx = episode_idx
        target_file = self.episode_files[episode_idx]
        print(f"🔄 [DASMCAPAdapter] 解析数据流: {target_file.name}")
        
        decoders: Dict[str, VideoDecoder] = {}
        temp_img_shape = None 
        
        with open(target_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, proto_msg in reader.iter_decoded_messages():
                topic = channel.topic
                
                # 1. 解析相机视频流 (基于 rules)
                if topic in self.camera_map:
                    std_cam_name = self.camera_map[topic]
                    
                    if std_cam_name not in self.images_cache:
                        self.images_cache[std_cam_name] = []
                        self.image_keys.append(std_cam_name)
                        decoders[std_cam_name] = VideoDecoder()
                        
                    img = decoders[std_cam_name].decode(proto_msg.data)
                    if img is not None:
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        # 无损保存
                        success, encoded_img = cv2.imencode('.png', img_bgr)
                        if success:
                            self.images_cache[std_cam_name].append(encoded_img.tobytes())
                            
                        if temp_img_shape is None:
                            temp_img_shape = img.shape[:2] 
                        if std_cam_name == self.image_keys[0]:
                            self.timestamps.append(message.publish_time)

                # 2. 解析相机内参
                elif "camera_info" in topic:
                    # 尝试从 topic 反推它属于哪个相机 (简单字符串匹配)
                    matching_cam = next((std_name for raw_topic, std_name in self.camera_map.items() if raw_topic.split('/')[1] == topic.split('/')[1]), None)
                    if matching_cam and matching_cam not in self.camera_info_cache:
                        width = getattr(proto_msg, 'width', temp_img_shape[1] if temp_img_shape else 1920)
                        height = getattr(proto_msg, 'height', temp_img_shape[0] if temp_img_shape else 1080)
                        self._extract_camera_info(matching_cam, proto_msg, width, height)

                # 3. 解析末端位姿 (基于 rules)
                elif topic in self.pose_map:
                    std_name = self.pose_map[topic]
                    pose = np.array([
                        proto_msg.pose.position.x, proto_msg.pose.position.y, proto_msg.pose.position.z,
                        proto_msg.pose.orientation.x, proto_msg.pose.orientation.y, proto_msg.pose.orientation.z, proto_msg.pose.orientation.w
                    ])
                    self.poses_cache[std_name].append((message.publish_time, pose))
                    
                # 4. 解析夹爪宽度 (基于 rules)
                elif topic in self.gripper_map:
                    std_name = self.gripper_map[topic]
                    # 假设pb定义中夹爪开合度的字段为 value (需根据实际 Protobuf 定义调整)
                    val = getattr(proto_msg, 'value', getattr(proto_msg, 'data', 0.0))
                    self.grippers_cache[std_name].append((message.publish_time, val))

        # 构建高频动作插值器
        self._build_interpolators()
        print(f"✅ [DASMCAPAdapter] Episode {episode_idx} 加载完毕，实际加载相机: {self.image_keys}")

    def _build_interpolators(self):
        """核心：构建位姿与夹爪的时序插值器，解决帧率不对齐问题"""
        self.interpolators.clear()
        
        # 处理位姿插值
        for arm_name, cache in self.poses_cache.items():
            if len(cache) < 2: continue
            
            # 去除重复的时间戳 (真实硬件数据常有此bug，会导致Slerp崩溃)
            times, unique_idx = np.unique([p[0] for p in cache], return_index=True)
            poses = np.array([p[1] for p in cache])[unique_idx]
            
            pos = poses[:, :3]
            rotations = st.Rotation.from_quat(poses[:, 3:7])
            
            # 位置线性插值，四元数球面插值
            self.interpolators[f"{arm_name}_pos"] = si.interp1d(times, pos, axis=0, bounds_error=False, fill_value=(pos[0], pos[-1]))
            self.interpolators[f"{arm_name}_rot"] = st.Slerp(times, rotations)
            
        # 处理夹爪插值
        for gripper_name, cache in self.grippers_cache.items():
            if len(cache) < 2: continue
            times, unique_idx = np.unique([g[0] for g in cache], return_index=True)
            vals = np.array([g[1] for g in cache])[unique_idx]
            
            self.interpolators[gripper_name] = si.interp1d(times, vals, bounds_error=False, fill_value=(vals[0], vals[-1]))

    def _extract_camera_info(self, std_cam_name: str, proto_msg, width: int, height: int):
        # ... (与你原代码一致，已省略以节省空间) ...
        pass

    def get_frame(self, index: int, specific_cameras: Optional[List[str]] = None) -> FrameData:
        if index < 0 or index >= len(self.timestamps): return None
        
        target_time = self.timestamps[index]
        images = {}
        camera_infos = {}
        keys_to_fetch = specific_cameras if specific_cameras is not None else self.image_keys
        
        for cam_name in keys_to_fetch:
            if cam_name in self.images_cache and index < len(self.images_cache[cam_name]):
                img_bytes = self.images_cache[cam_name][index]
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                decoded_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if decoded_img is not None:
                    img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
                    if cam_name in self.camera_info_cache:
                        info = self.camera_info_cache[cam_name]
                        if self.enable_undistort and 'undistort_maps' in info:
                            map_x, map_y = info['undistort_maps']
                            img = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
                        camera_infos[cam_name] = {k: v for k, v in info.items() if k != 'undistort_maps'}
                    images[cam_name] = img
        
        # === 构建高精度的插值 QPos 向量 ===
        # 标准输出顺序: 按配置的 key 字母序拼接 (比如先 left_ee, 再 right_ee, 再夹爪)
        qpos_list = []
        
        for arm_name in sorted(set(self.pose_map.values())):
            if f"{arm_name}_pos" in self.interpolators:
                exact_pos = self.interpolators[f"{arm_name}_pos"](target_time)
                exact_rot = self.interpolators[f"{arm_name}_rot"](target_time).as_quat()
                qpos_list.extend(exact_pos.tolist())
                qpos_list.extend(exact_rot.tolist())
                
        for gripper_name in sorted(set(self.gripper_map.values())):
            if gripper_name in self.interpolators:
                exact_grip = self.interpolators[gripper_name](target_time)
                qpos_list.append(float(exact_grip))

        state = {}
        if qpos_list:
            state['qpos'] = np.array(qpos_list)
        if camera_infos:
            state['camera_infos'] = camera_infos
            
        return FrameData(
            timestamp=target_time/1e9, 
            images=images, 
            state=state
        )

    def close(self):
        self.images_cache.clear()
        self.camera_info_cache.clear()
        self.timestamps.clear()
        for k in self.poses_cache: self.poses_cache[k].clear()
        for k in self.grippers_cache: self.grippers_cache[k].clear()
        self.interpolators.clear()
        self.image_keys.clear()