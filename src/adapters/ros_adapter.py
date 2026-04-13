# src/adapters/ros_adapter.py
from pathlib import Path
import numpy as np
import cv2
from typing import List, Dict, Any, Optional

from mcap.reader import make_reader
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from src.core.interface import BaseDatasetReader, FrameData, AdapterConfig
from src.core.registry import AdapterRegistry

@AdapterRegistry.register("ROS")
class RosAdapter(BaseDatasetReader):
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.root_path = None
        self.reader = None
        self.is_mcap = False
        
        # 1. 基础配置标准化
        self.camera_map = getattr(self.config, 'image_keys_map', {}) or {}
        self.arm_groups = getattr(self.config, 'arm_groups', {}) or {}
        self.base_map = getattr(self.config, 'state_keys_map', {}) or {}
        
        extra_opts = getattr(self.config, 'extra_options', {}) or {}
        self.ignore_topics = extra_opts.get("ignore_topics", [])
        
        self.image_topics = []
        self.timestamps = []
        self.mcap_messages = []
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)
        self._length = 0
        
        self.episode_files = []
        self.current_episode_idx = 0

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        self.episode_files = []
        
        if self.root_path.is_file() and self.root_path.suffix.lower() in ['.mcap', '.bag']:
            self.episode_files.append(self.root_path)
        elif self.root_path.is_dir():
            self.episode_files.extend(sorted(self.root_path.rglob("*.mcap")))
            self.episode_files.extend(sorted(self.root_path.rglob("*.bag")))
            
        if not self.episode_files: return False
            
        print(f"✅ [ROS] 扫描到 {len(self.episode_files)} 个数据包")
        self.set_episode(0)
        return True

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episode_files): return
        self.current_episode_idx = episode_idx
        self.close()
        
        target_file = self.episode_files[episode_idx]
        str_path = str(target_file.absolute())
        
        self.mcap_messages = []
        self.image_topics = []
        self.timestamps = []
        self._length = 0
        all_found_topics = {}

        try:
            if target_file.suffix.lower() == '.mcap':
                self.is_mcap = True
                with open(str_path, "rb") as f:
                    reader = make_reader(f)
                    for schema, channel, message in reader.iter_messages():
                        topic_name = channel.topic
                        msg_type = schema.name if schema else "Unknown"
                        if topic_name not in all_found_topics:
                            all_found_topics[topic_name] = msg_type
                        
                        if 'image' in topic_name.lower() or 'image' in msg_type.lower():
                            self.mcap_messages.append({'topic': topic_name, 'publish_time': message.publish_time, 'data': message.data, 'msgtype': msg_type})
                self.image_topics = [t for t in all_found_topics.keys() if 'image' in t.lower()]
            else:
                self.is_mcap = False
                self.reader = AnyReader([target_file], default_typestore=self.typestore)
                self.reader.open() 
                self.image_topics = [c.topic for c in self.reader.connections if 'Image' in c.msgtype]

            # 应用过滤与 Config 的映射关系
            if self.ignore_topics:
                self.image_topics = [t for t in self.image_topics if not any(kw in t.lower() for kw in self.ignore_topics)]

            if self.camera_map:
                target_topics = list(self.camera_map.values())
                self.image_topics = [t for t in self.image_topics if t in target_topics or f"/{t}" in target_topics]

            if not self.image_topics: return
            
            primary = self.image_topics[0]
            if self.is_mcap:
                self.timestamps = sorted([m['publish_time'] for m in self.mcap_messages if m['topic'] == primary])
            else:
                conns = [c for c in self.reader.connections if c.topic == primary]
                self.timestamps = sorted([ts for _, ts, _ in self.reader.messages(connections=conns)])
                
            self._length = len(self.timestamps)
        except Exception as e:
            print(f"🚨 [ROS 警告] 轨迹加载失败: {e}")
            self.close()

    def get_total_episodes(self) -> int: return len(self.episode_files)
    def get_length(self) -> int: return self._length

    def get_all_sensors(self) -> List[str]:
        if self.camera_map: return list(self.camera_map.keys())
        return [t.lstrip('/') for t in self.image_topics]

    def _get_standard_cam_name(self, original_topic: str) -> str:
        std_cam_name = original_topic.lstrip('/')
        if self.camera_map:
            for k, v in self.camera_map.items():
                if v == original_topic or v == original_topic.lstrip('/'):
                    std_cam_name = k
                    break
        return std_cam_name

    def get_frame(self, index: int, specific_cameras: Optional[List[str]] = None) -> FrameData:
        if index < 0 or index >= self._length: return None
        target_time = self.timestamps[index]
        window = 50 * 10**6
        images = {}

        keys_to_fetch = specific_cameras if specific_cameras else self.get_all_sensors()
        # 反查允许解析的原始 topics
        allowed_topics = []
        if self.camera_map:
            for k in keys_to_fetch:
                if k in self.camera_map: allowed_topics.extend([self.camera_map[k], f"/{self.camera_map[k]}"])
        else:
            allowed_topics = [t for t in self.image_topics if t.lstrip('/') in keys_to_fetch]

        if self.is_mcap:
            for m in self.mcap_messages:
                if m['topic'] in allowed_topics and abs(m['publish_time'] - target_time) < window:
                    try:
                        msg = self.typestore.deserialize_cdr(m['data'], m['msgtype'])
                        img = self._process_ros_msg(msg)
                        if img is not None: images[self._get_standard_cam_name(m['topic'])] = img
                    except Exception: pass
        else:
            conns = [c for c in self.reader.connections if c.topic in allowed_topics]
            for conn, ts, rawdata in self.reader.messages(connections=conns, start=target_time-window, stop=target_time+window):
                try:
                    msg = self.reader.deserialize(rawdata, conn.msgtype)
                    img = self._process_ros_msg(msg)
                    if img is not None: images[self._get_standard_cam_name(conn.topic)] = img
                except Exception: pass
        
        return FrameData(timestamp=float(target_time)/1e9, images=images, state={})

    def _process_ros_msg(self, msg) -> np.ndarray:
        try:
            img_raw = np.frombuffer(msg.data, dtype=np.uint8)
            if hasattr(msg, 'format'):
                frame = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
            h, w = msg.height, msg.width
            encoding = getattr(msg, 'encoding', 'rgb8').lower()
            if '16' in encoding: img_raw = np.frombuffer(msg.data, dtype=np.uint16)
            if 'bayer' in encoding: return cv2.cvtColor(img_raw.reshape(h, w), cv2.COLOR_BayerBG2RGB)
            elif 'rgb' in encoding: return img_raw.reshape(h, w, 3)
            elif 'bgr' in encoding: return cv2.cvtColor(img_raw.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
            elif 'mono' in encoding: return img_raw.reshape(h, w)
            else: return img_raw.reshape(h, w, -1)
        except: return None

    def get_current_episode_path(self) -> str:
        if self.episode_files and 0 <= self.current_episode_idx < len(self.episode_files): return str(self.episode_files[self.current_episode_idx])
        return None

    def close(self):
        if self.reader:
            try: self.reader.close()
            except: pass
            self.reader = None