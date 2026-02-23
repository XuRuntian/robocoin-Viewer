"""
# Robo-ETL 交互式数据对齐工作台

基于 Streamlit 实现的可视化数据校验工具，支持人工微调 AI 预标注的动作切分边界
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.core.factory import ReaderFactory
import numpy as np
import json
import os

def generate_mock_subtasks(total_frames):
    """生成模拟的AI预标注数据（包含尖峰特征的典型任务分布）"""
    return [
        {
            "subtask_id": "grasp_001",
            "instruction": "夹爪闭合抓取物体",
            "start_frame": int(total_frames * 0.2),
            "end_frame": int(total_frames * 0.35)
        },
        {
            "subtask_id": "lift_002",
            "instruction": "垂直提升物体",
            "start_frame": int(total_frames * 0.35),
            "end_frame": int(total_frames * 0.55)
        },
        {
            "subtask_id": "place_003",
            "instruction": "放置物体到目标位置",
            "start_frame": int(total_frames * 0.55),
            "end_frame": int(total_frames * 0.8)
        }
    ]

def generate_mock_waveform(total_frames):
    """生成带尖峰的随机机械臂运动轨迹数据（复合速度L2 Norm）"""
    base = np.sin(np.linspace(0, 4*np.pi, total_frames)) * 0.5
    spikes = np.zeros(total_frames)
    spikes[np.random.choice(total_frames, size=5, replace=False)] = 2.0
    return base + spikes + np.random.normal(0, 0.1, total_frames)

def main():
    """主函数：构建Streamlit界面并处理用户交互"""
    st.set_page_config(page_title="Robo-ETL 数据对齐工作台", layout="wide")
    
    # 1. 数据源与顶层控制 (侧边栏)
    with st.sidebar:
        st.header("数据源配置")
        data_path = st.text_input("HDF5文件路径", value="/data/robot_demos/demo.hdf5")
        
        if st.button("加载数据"):
            try:
                # 初始化Reader
                reader = ReaderFactory.get_reader(data_path)
                total_frames = reader.get_total_frames()
                
                # 读取或生成数据
                try:
                    # 尝试使用Reader的原生方法
                    subtasks = reader.get_subtasks()
                    waveform = reader.get_kinematic_data()
                except AttributeError:
                    # 使用Mock数据作为后备
                    subtasks = generate_mock_subtasks(total_frames)
                    waveform = generate_mock_waveform(total_frames)
                
                # 存储在session_state中供后续使用
                st.session_state.update({
                    'reader': reader,
                    'total_frames': total_frames,
                    'subtasks': subtasks,
                    'waveform': waveform,
                    'current_frame': 0
                })
                
                st.success("数据加载成功！")
            except Exception as e:
                st.error(f"数据加载失败：{str(e)}")

    # 2. 视频/图像预览区 (左侧布局)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("图像预览")
        if 'reader' in st.session_state:
            current_frame = st.slider(
                "选择帧号", 
                0, 
                st.session_state.total_frames-1,
                key='frame_slider',
                on_change=lambda: st.session_state.update({'current_frame': st.session_state.frame_slider})
            )
            
            try:
                frame = st.session_state.reader.get_frame(current_frame)
                st.image(frame, caption=f"帧 {current_frame}", use_column_width=True)
            except Exception as e:
                st.error(f"图像读取失败：{str(e)}")
        else:
            st.info("请先加载数据集")

    # 3. 多模态时间轴与物理波形图 (右侧布局)
    with col2:
        st.subheader("时间轴与运动学分析")
        
        if 'subtasks' in st.session_state:
            # 创建联动图表
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            
            # 上层：时间轴（甘特图）
            for i, task in enumerate(st.session_state.subtasks):
                fig.add_shape(
                    type="rect",
                    x0=task['start_frame'], x1=task['end_frame'],
                    y0=0, y1=1,
                    fillcolor=f"rgba({i*80}, 150, 200, 0.6)",
                    line_width=0,
                    row=1, col=1,
                    hoverinfo="text",
                    text=f"{task['instruction']}<br>帧: {task['start_frame']}-{task['end_frame']}"
                )
            
            # 下层：波形图
            fig.add_trace(
                go.Scatter(
                    x=np.arange(st.session_state.total_frames),
                    y=st.session_state.waveform,
                    mode='lines',
                    name='运动学特征',
                    line=dict(color='rgb(55, 126, 184)')
                ),
                row=2, col=1
            )
            
            # 图表布局设置
            fig.update_layout(
                height=600,
                showlegend=False,
                xaxis2_title="帧号",
                yaxis=dict(title="任务", showticklabels=False, showgrid=False),
                yaxis2=dict(title="复合速度 (L2 Norm)")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请先加载数据集")

    # 4. 交互式数据对齐编辑器 (底部布局)
    st.subheader("预标注编辑器")
    
    if 'subtasks' in st.session_state:
        # 使用st.data_editor进行表格编辑
        edited_subtasks = st.data_editor(
            st.session_state.subtasks,
            column_config={
                "subtask_id": st.column_config.TextColumn("任务ID", disabled=True),
                "instruction": st.column_config.TextColumn("自然语言指令"),
                "start_frame": st.column_config.NumberColumn("起始帧", min_value=0),
                "end_frame": st.column_config.NumberColumn("结束帧", min_value=0)
            },
            num_rows="dynamic",
            key='subtask_editor'
        )
        
        # 当表格数据变化时更新session_state
        if edited_subtasks != st.session_state.subtasks:
            st.session_state.subtasks = edited_subtasks
            st.experimental_rerun()  # 强制刷新以更新图表
    
    # 5. 导出功能
    if 'subtasks' in st.session_state:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("保存修改"):
                try:
                    # 这里可以添加持久化保存逻辑
                    st.success("修改已保存到内存")
                except Exception as e:
                    st.error(f"保存失败：{str(e)}")
        
        with col2:
            json_data = json.dumps(st.session_state.subtasks, indent=2)
            st.download_button(
                label="导出JSON",
                data=json_data,
                file_name="aligned_subtasks.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
