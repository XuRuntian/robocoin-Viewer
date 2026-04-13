from dataclasses import dataclass
import numpy as np

@dataclass
class ArmState:
    """机械臂状态的标准协议"""
    pos: np.ndarray      # 笛卡尔坐标 (T, 3)
    rot: np.ndarray      # 旋转欧拉角 (T, 3)
    gripper: np.ndarray  # 夹爪状态 (T, 1) 或 (T, 2)
    arm_type: str            # "right" 或 "left"
    
    def __post_init__(self):
        """数据维度和类型强制清洗防火墙"""
        # 1. 统一类型为 float32 (对齐深度学习和物理计算)
        self.pos = np.asarray(self.pos, dtype=np.float32)
        self.rot = np.asarray(self.rot, dtype=np.float32)
        self.gripper = np.asarray(self.gripper, dtype=np.float32)

        # 2. 强制转换维度为 (T, D)
        # 如果传进来的是 1D 数组 (T,)，自动升维变成 (T, 1)
        if self.pos.ndim == 1: self.pos = self.pos.reshape(-1, 1)
        if self.rot.ndim == 1: self.rot = self.rot.reshape(-1, 1)
        if self.gripper.ndim == 1: self.gripper = self.gripper.reshape(-1, 1)
        
        # 如果传进来的是 3D 数组比如 (1, 2, T) 或者 (T, 1, 2)，强制压平成 (T, D)
        # 假设第 0 轴永远是时间轴 T
        T = self.pos.shape[0]
        self.pos = self.pos.reshape(T, -1)
        self.rot = self.rot.reshape(T, -1)
        self.gripper = self.gripper.reshape(T, -1)