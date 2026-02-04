# src/ui/rerun_visualizer.py
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from typing import List
from src.core.interface import FrameData

class RerunVisualizer:
    def __init__(self, app_name: str = "RoboCoin_Viewer"):
        self.app_name = app_name
        rr.init(self.app_name, spawn=True)
        # 注意：这里不再自动调用 _setup_blueprint
        # 我们等待外部传入相机列表后再初始化布局

    def setup_layout(self, camera_names: List[str]):
        """
        根据传入的相机列表，动态生成 Rerun 布局
        """
        print(f"[Rerun] 初始化布局，检测到相机: {camera_names}")

        # 1. 动态生成相机视图列表
        # 我们把每个相机都创建一个 Spatial2DView
        cam_views = []
        for cam_name in camera_names:
            cam_views.append(
                rrb.Spatial2DView(
                    name=cam_name,  # 视图标题就是相机名
                    origin=f"world/camera/{cam_name}" # 数据路径
                )
            )

        # 2. 组装 Blueprint
        # 使用 Grid 布局自动排列所有相机，下方放一个状态波形图
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                # 上半部分：相机网格 (Grid 会自动处理 1个、3个或 N个的情况)
                rrb.Grid(
                    *cam_views, 
                    grid_columns=2 if len(camera_names) > 1 else 1 # 如果相机多，就双列显示
                ),
                # 下半部分：关节状态
                rrb.TimeSeriesView(name="Joint States", origin="world/robot/qpos"),
                row_shares=[3, 1] # 相机占 3 份高度，波形图占 1 份
            ),
            collapse_panels=True
        )
        
        # 发送布局给 Rerun Viewer
        rr.send_blueprint(blueprint)

    def log_frame(self, frame: FrameData, frame_idx: int):
        """将标准数据帧推送到 Rerun"""
        rr.set_time_sequence("frame_idx", frame_idx)
        rr.set_time_seconds("log_time", frame.timestamp)

        # 动态 Log 所有相机
        for cam_name, img in frame.images.items():
            rr.log(f"world/camera/{cam_name}", rr.Image(img))

        # Log 关节状态
        if frame.state and 'qpos' in frame.state:
            qpos = frame.state['qpos']
            for i, val in enumerate(qpos):
                rr.log(f"world/robot/qpos/j{i}", rr.Scalars(val))