from pathlib import Path
import numpy as np
import cv2
import traceback
from typing import List, Dict, Any

# 核心：底层 mcap 读取
from mcap.reader import make_reader
# rosbags 负责处理 ROS1 和 CDR 解码
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
# 导入接口
from src.core.interface import BaseDatasetReader, FrameData

class RosAdapter(BaseDatasetReader):
    def __init__(self):
        self.reader = None
        self.is_mcap = False
        self.image_topics = []
        self.timestamps = []
        self.mcap_messages = []  # 存储扫描到的原始消息
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)
        self._length = 0

    def load(self, file_path: str) -> bool:
        try:
            p = Path(file_path)
            str_path = str(p.absolute())
            self.mcap_messages = []
            all_found_topics = {}

            if p.suffix.lower() == '.mcap':
                print(f"[MCAP] 扫描文件: {str_path} ({p.stat().st_size/1024/1024:.2f}MB)")
                self.is_mcap = True
                with open(str_path, "rb") as f:
                    reader = make_reader(f)
                    # 暴力扫描所有消息
                    for schema, channel, message in reader.iter_messages():
                        topic_name = channel.topic
                        msg_type = schema.name if schema else "Unknown"
                        
                        if topic_name not in all_found_topics:
                            all_found_topics[topic_name] = msg_type
                        
                        # 只缓存图像数据到内存字典中（如果是 TB 级大数据，建议后期改为按需索引）
                        if 'image' in topic_name.lower() or 'image' in msg_type.lower():
                            self.mcap_messages.append({
                                'topic': topic_name,
                                'publish_time': message.publish_time,
                                'data': message.data,
                                'msgtype': msg_type
                            })

                self.image_topics = [t for t in all_found_topics.keys() if 'image' in t.lower()]
                if not self.image_topics:
                    print(f"❌ 未发现图像 Topic。所有发现: {list(all_found_topics.keys())}")
                    return False
                
                # 建立统一索引
                primary = self.image_topics[0]
                self.timestamps = sorted([m['publish_time'] for m in self.mcap_messages if m['topic'] == primary])
                self._length = len(self.timestamps)
                print(f"[ROS] 加载成功: 发现 {len(all_found_topics)} 个 Topic, 图像帧数 {self._length}")
                return True

            else:
                # ROS1 (.bag) 逻辑
                print(f"[ROS] 使用 AnyReader 加载 ROS1: {p.name}")
                self.is_mcap = False
                self.reader = AnyReader([p], default_typestore=self.typestore)
                self.reader.open()
                self.image_topics = [c.topic for c in self.reader.connections if 'Image' in c.msgtype]
                if not self.image_topics: return False
                
                primary = self.image_topics[0]
                conns = [c for c in self.reader.connections if c.topic == primary]
                self.timestamps = sorted([ts for _, ts, _ in self.reader.messages(connections=conns)])
                self._length = len(self.timestamps)
                return True

        except Exception:
            traceback.print_exc()
            return False

    def get_length(self) -> int:
        return self._length

    def get_all_sensors(self) -> List[str]:
        # 返回去重后的相机/Topic 名字 (去掉开头的 /)
        return [t.lstrip('/') for t in self.image_topics]

    def get_frame(self, index: int) -> FrameData:
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of bounds")

        target_time = self.timestamps[index]
        window = 50 * 10**6  # 50ms 窗口同步
        images = {}

        if self.is_mcap:
            # 搜索匹配时间的帧
            for m in self.mcap_messages:
                if abs(m['publish_time'] - target_time) < window:
                    try:
                        msg = self.typestore.deserialize_cdr(m['data'], m['msgtype'])
                        img = self._process_ros_msg(msg)
                        if img is not None:
                            images[m['topic'].lstrip('/')] = img
                    except:
                        continue
        else:
            conns = [c for c in self.reader.connections if c.topic in self.image_topics]
            for conn, ts, rawdata in self.reader.messages(connections=conns, 
                                                         start=target_time-window, 
                                                         stop=target_time+window):
                msg = self.typestore.deserialize_cdr(rawdata, conn.msgtype)
                img = self._process_ros_msg(msg)
                if img is not None:
                    images[conn.topic.lstrip('/')] = img
        
        # ROS 暂时不处理 state，可以留空
        return FrameData(
            timestamp=float(target_time) / 1e9, # 转为秒
            images=images,
            state={}
        )

    def _process_ros_msg(self, msg) -> np.ndarray:
        """ 图像解码核心逻辑 """
        try:
            img_raw = np.frombuffer(msg.data, dtype=np.uint8)
            # 压缩图像处理
            if hasattr(msg, 'format'):
                frame = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
            
            # 原始图像处理
            h, w = msg.height, msg.width
            encoding = getattr(msg, 'encoding', 'rgb8').lower()
            
            if '16' in encoding:
                img_raw = np.frombuffer(msg.data, dtype=np.uint16)

            if 'bayer' in encoding:
                frame = img_raw.reshape(h, w)
                return cv2.cvtColor(frame, cv2.COLOR_BayerBG2RGB)
            elif 'rgb' in encoding:
                return img_raw.reshape(h, w, 3)
            elif 'bgr' in encoding:
                frame = img_raw.reshape(h, w, 3)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif 'mono' in encoding:
                return img_raw.reshape(h, w)
            else:
                return img_raw.reshape(h, w, -1)
        except:
            return None

    def close(self):
        if self.reader:
            self.reader.close()