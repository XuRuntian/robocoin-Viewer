# src/adapters/ego_adapter.py
import os
import av
import cv2
import numpy as np
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

@AdapterRegistry.register("Ego")
class EgoAdapter(BaseDatasetReader):
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.root_path = None
        self.episode_files = []
        self.current_episode_idx = 0
        
        self.enable_undistort = False
        if self.config and self.config.extra_options:
            self.enable_undistort = self.config.extra_options.get("enable_undistort", False)
        
        self.image_keys = []           
        self.timestamps = []           
        self.images_cache = {}         
        self.camera_info_cache = {}    
        self.pose_cache = []           

    @staticmethod
    def _generate_ds_map_numerical(width, height, fu, fv, cu, cv, xi, alpha):
        u_out, v_out = np.meshgrid(np.arange(width), np.arange(height))
        u_out = u_out.astype(np.float64)
        v_out = v_out.astype(np.float64)
        
        u_out_f = u_out.astype(np.float32)
        v_out_f = v_out.astype(np.float32)
        
        x = (u_out_f - cu) / fu
        y = (v_out_f - cv) / fv
        z = np.ones_like(x)
        
        d1 = np.sqrt(x*x + y*y + z*z)
        mz = (1.0 - alpha) * d1 + alpha * z
        d2 = np.sqrt(x*x + y*y + mz*mz)
        
        denominator = (1.0 - alpha) * d2 + alpha * (xi * d1 + z)
        denominator = np.clip(denominator, 1e-8, None)
        
        u_in = fu * x / denominator + cu
        v_in = fv * y / denominator + cv
        
        return u_in.astype(np.float32), v_in.astype(np.float32)

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        self.episode_files = []
        
        if self.root_path.is_file() and self.root_path.suffix.lower() == '.mcap':
            self.episode_files.append(self.root_path)
        elif self.root_path.is_dir():
            self.episode_files.extend(sorted(self.root_path.rglob("*.mcap")))
            
        if not self.episode_files:
            print("❌ [EgoAdapter] 未找到任何 .mcap 文件。")
            return False
            
        print(f"✅ [EgoAdapter] 扫描到 {len(self.episode_files)} 个 DAS Protobuf MCAP 数据包")
        self.set_episode(0)
        return True

    def _get_standard_cam_name(self, original_cam_name: str) -> Optional[str]:
        if not self.config or not self.config.image_keys_map:
            return original_cam_name 
            
        for std_name, orig_rule in self.config.image_keys_map.items():
            if orig_rule == original_cam_name or original_cam_name in orig_rule:
                return std_name
        return None

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episode_files): return
        
        self.close()  
        self.current_episode_idx = episode_idx
        target_file = self.episode_files[episode_idx]
        
        print(f"🔄 [EgoAdapter] 正在解析并解码数据流: {target_file.name}")
        
        decoders: Dict[str, VideoDecoder] = {}
        temp_img_shape = None 
        
        with open(target_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, proto_msg in reader.iter_decoded_messages():
                topic = channel.topic
                
                # 1. 解析相机视频流
                if "/sensor/" in topic and "/compressed" in topic:
                    raw_cam_name = topic.split('/')[-2]
                    std_cam_name = self._get_standard_cam_name(raw_cam_name)
                    
                    if not std_cam_name:
                        continue 
                        
                    if std_cam_name not in self.images_cache:
                        self.images_cache[std_cam_name] = []
                        self.image_keys.append(std_cam_name)
                        decoders[std_cam_name] = VideoDecoder()
                        
                    # ⚠️ 核心修复 1：H264连续解码出原始数组
                    img = decoders[std_cam_name].decode(proto_msg.data)
                    
                    if img is not None:
                        # ⚠️ 核心修复 2：立即将体积巨大的 Numpy 转为轻量级 JPEG Bytes 存入内存
                        # img 是 rgb24，cv2 需要 bgr
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        # 质量设为 85，肉眼几乎无损，体积缩减 30 倍
                        success, encoded_img = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        
                        if success:
                            self.images_cache[std_cam_name].append(encoded_img.tobytes())
                            
                        if temp_img_shape is None:
                            temp_img_shape = img.shape[:2] 
                            
                        if std_cam_name == self.image_keys[0]:
                            self.timestamps.append(message.publish_time)
                            
                # 2. 解析相机内参
                elif "camera_info" in topic:
                    raw_cam_name = topic.split('/')[-2]
                    std_cam_name = self._get_standard_cam_name(raw_cam_name)
                    
                    if not std_cam_name:
                        continue
                        
                    if std_cam_name not in self.camera_info_cache:
                        width = getattr(proto_msg, 'width', temp_img_shape[1] if temp_img_shape else 1920)
                        height = getattr(proto_msg, 'height', temp_img_shape[0] if temp_img_shape else 1080)
                        self._extract_camera_info(std_cam_name, proto_msg, width, height)

                # 3. 解析头部 VIO 位姿
                elif topic == "/robot0/vio/eef_pose":
                    pose = np.array([
                        proto_msg.pose.position.x, proto_msg.pose.position.y, proto_msg.pose.position.z,
                        proto_msg.pose.orientation.x, proto_msg.pose.orientation.y, proto_msg.pose.orientation.z, proto_msg.pose.orientation.w
                    ])
                    self.pose_cache.append((message.publish_time, pose))

        self.pose_cache.sort(key=lambda x: x[0])
        print(f"✅ [EgoAdapter] Episode {episode_idx} 加载完毕，实际加载相机: {self.image_keys}")

    def _extract_camera_info(self, std_cam_name: str, proto_msg, width: int, height: int):
        k_matrix = getattr(proto_msg, 'K', getattr(proto_msg, 'k', None))
        d_coeffs = getattr(proto_msg, 'D', getattr(proto_msg, 'd', None))
        
        if k_matrix and len(k_matrix) >= 9 and d_coeffs:
            K = np.array(k_matrix, dtype=np.float64).reshape(3, 3)
            D = np.array(d_coeffs, dtype=np.float64)
            model = getattr(proto_msg, 'distortion_model', 'unknown').lower()
            
            info = {'K': K, 'D': D, 'model': model}
            
            if self.enable_undistort and (model == 'ds' or len(D) == 6):
                fu, fv, cu, cv = K[0,0], K[1,1], K[0,2], K[1,2]
                xi, alpha = D[4], D[5] if len(D) > 5 else 0.0
                print(f"⚙️ 正在为 {std_cam_name} 计算数值法去畸变映射表...")
                map_x, map_y = self._generate_ds_map_numerical(width, height, fu, fv, cu, cv, xi, alpha)
                info['undistort_maps'] = (map_x, map_y)
                info['D'] = np.zeros_like(D)
                info['model'] = 'none'
                
            self.camera_info_cache[std_cam_name] = info
            print(f"📸 [EgoAdapter] 已记录相机内参: {std_cam_name} (最终模型: {info['model']})")

    def get_total_episodes(self) -> int:
        return len(self.episode_files)

    def get_length(self) -> int:
        return len(self.timestamps)

    def get_all_sensors(self) -> List[str]:
        return self.image_keys

    def get_current_episode_path(self) -> str:
        if self.episode_files and 0 <= self.current_episode_idx < len(self.episode_files):
            return str(self.episode_files[self.current_episode_idx])
        return None

    def get_frame(self, index: int, specific_cameras: Optional[List[str]] = None) -> FrameData:
        if index < 0 or index >= len(self.timestamps): return None
        
        target_time = self.timestamps[index]
        images = {}
        camera_infos = {}
        keys_to_fetch = specific_cameras if specific_cameras is not None else self.image_keys
        
        for cam_name in keys_to_fetch:
            if cam_name in self.images_cache and index < len(self.images_cache[cam_name]):
                # ⚠️ 核心修复 3：从内存中取出轻量级的 JPEG bytes 并瞬间还原为图片
                img_bytes = self.images_cache[cam_name][index]
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                decoded_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if decoded_img is not None:
                    # 恢复为标准的 RGB 通道
                    img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
                    
                    # 应用去畸变 (如果开启)
                    if cam_name in self.camera_info_cache:
                        info = self.camera_info_cache[cam_name]
                        if self.enable_undistort and 'undistort_maps' in info:
                            map_x, map_y = info['undistort_maps']
                            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                        camera_infos[cam_name] = {k: v for k, v in info.items() if k != 'undistort_maps'}
                    
                    images[cam_name] = img
        
        qpos_list = []
        if self.pose_cache:
            idx = np.searchsorted([p[0] for p in self.pose_cache], target_time)
            idx = min(idx, len(self.pose_cache)-1)
            qpos_list.extend(self.pose_cache[idx][1])

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
        self.pose_cache.clear()
        self.image_keys.clear()