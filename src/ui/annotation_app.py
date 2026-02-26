import sys
import os
import json
import time
import streamlit as st
import rerun as rr
import rerun.blueprint as rrb

# 确保能找到 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.inspector import DatasetInspector
from src.core.organizer import DatasetOrganizer
from src.core.reviewer import DatasetReviewer
from src.core.factory import ReaderFactory
from src.core.config_generator import ConfigGenerator
from src.ui.rerun_visualizer import RerunVisualizer

# 页面配置
st.set_page_config(page_title="RoboCoin Annotation Tool", layout="wide")

@st.cache_data
def load_vocabulary():
    """读取外部词库文件 (Schema 配置)"""
    vocab_path = os.path.join(os.path.dirname(__file__), '../../configs/vocabulary.json')
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"找不到词库文件: {vocab_path}")
        return {"fields": []}

def clean_editor_value(val):
    """
    从 'English (中文)' 格式中提取 'English'
    用于清洗 data_editor 的返回值
    """
    if isinstance(val, str) and " (" in val and val.endswith(")"):
        return val.split(" (")[0]
    return val

def render_field(field, current_data):
    """配置驱动：根据 schema 定义自动渲染 Streamlit 控件并返回收集到的值"""
    key = field["key"]
    label = field["label"]
    ftype = field["type"]
    
    if ftype == "text":
        return st.text_input(label, placeholder=field.get("placeholder", ""))
        
    elif ftype == "textarea":
        val = st.text_area(label, value=field.get("default", ""), height=100)
        # 自动将文本框多行文字转换为 List
        return [i.strip() for i in val.split('\n') if i.strip()] 
        
    elif ftype == "selectbox":
        opts_dict = field.get("options", {})
        opts_keys = list(opts_dict.keys())
        fmt_func = lambda x, d=opts_dict: f"{x} ({d[x]})" if d.get(x) else x
        return st.selectbox(label, options=opts_keys, format_func=fmt_func)

    # 👇 新增的级联下拉框 (Dependent Selectbox) 逻辑 👇
    elif ftype == "selectbox_dependent":
        parent_key = field.get("depends_on")
        # 去当前已渲染的数据里找父级的值
        parent_val = current_data.get(parent_key) 
        
        opts_map = field.get("options_map", {})
        # 根据父级的值，从 map 中获取对应的子选项字典。如果没找到，给个防崩的默认提示
        opts_dict = opts_map.get(parent_val, {"unknown": "请先选择上级或暂无选项"})
        
        opts_keys = list(opts_dict.keys())
        fmt_func = lambda x, d=opts_dict: f"{x} ({d[x]})" if d.get(x) and x != "unknown" else x
        return st.selectbox(label, options=opts_keys, format_func=fmt_func)
    # 👆 --------------------------------- 👆
        
    elif ftype == "multiselect":
        opts_dict = field.get("options", {})
        opts_keys = list(opts_dict.keys())
        fmt_func = lambda x, d=opts_dict: f"{x} ({d[x]})" if d.get(x) else x
        return st.multiselect(label, options=opts_keys, format_func=fmt_func)
        
    elif ftype == "number":
        return st.number_input(label, value=field.get("default", 0.0), format="%.1f")
        
    elif ftype == "object_table":
        # 专门处理复杂表格
        if f'table_{key}' not in st.session_state:
            st.session_state[f'table_{key}'] = [{"object_name": "table", "color": "red"}]
            
        name_opts = field.get("name_options", {})
        color_opts = field.get("color_options", {})
        name_display = [f"{k} ({v})" for k, v in name_opts.items()]
        color_display = [f"{k} ({v})" for k, v in color_opts.items()]
        
        edited = st.data_editor(
            st.session_state[f'table_{key}'],
            num_rows="dynamic",
            column_config={
                "object_name": st.column_config.SelectboxColumn("物品名称", options=name_display, required=True),
                "color": st.column_config.SelectboxColumn("颜色", options=color_display, required=True)
            },
            width="stretch" # 替代废弃的 use_container_width
        )
        
        # 自动清洗表格里的中文标签
        cleaned = []
        for row in edited:
            if row.get("object_name"):
                cleaned.append({
                    "object_name": clean_editor_value(row["object_name"]),
                    "color": clean_editor_value(row["color"])
                })
        return cleaned

def setup_comparison_layout(sample_names, cameras):
    """配置 Rerun 并排对比视图的蓝图"""
    columns = []
    for idx, name in enumerate(sample_names):
        cam_views = []
        for cam in cameras:
            cam_views.append(rrb.Spatial2DView(
                origin=f"preview/sample_{idx}/{cam}",
                name=f"{cam}"
            ))
        columns.append(rrb.Vertical(
            rrb.TextDocumentView(origin=f"preview/sample_{idx}/info", name=f"{name}"),
            *cam_views,
            name=f"Sample {idx+1}"
        ))
    blueprint = rrb.Blueprint(rrb.Horizontal(*columns), collapse_panels=True)
    rr.send_blueprint(blueprint)

def run_parallel_preview(sample_paths):
    """执行多样本抽样预览播放逻辑"""
    temp_reader = ReaderFactory.get_reader(sample_paths[0])
    temp_reader.load(sample_paths[0])
    cameras = temp_reader.get_all_sensors()
    
    rr.init("RoboCoin_Preview", spawn=True)
    setup_comparison_layout([os.path.basename(p) for p in sample_paths], cameras)
    temp_reader.close()
    
    rr.log("preview", rr.Clear(recursive=True))
    
    readers = []
    max_len = 0
    for p in sample_paths:
        r = ReaderFactory.get_reader(p)
        r.load(p)
        readers.append(r)
        if r.get_length() > max_len: 
            max_len = r.get_length()
    
    progress_bar = st.progress(0, text="正在同步播放视频流...")
    for i in range(max_len):
        rr.set_time_sequence("frame_idx", i)
        for s_idx, r in enumerate(readers):
            if i >= r.get_length(): continue
            frame = r.get_frame(i)
            for cam, img in frame.images.items():
                rr.log(f"preview/sample_{s_idx}/{cam}", rr.Image(img))
            if i == 0:
                rr.log(f"preview/sample_{s_idx}/info", rr.TextDocument(f"### {os.path.basename(sample_paths[s_idx])}"))
        
        # UI 进度条更新
        if i % 10 == 0 or i == max_len - 1:
            progress_bar.progress((i + 1) / max_len, text=f"播放进度: {i+1}/{max_len} 帧")
            
    for r in readers: r.close()
    progress_bar.empty()
    st.success("✅ 预览播放完成，请在 Rerun 窗口查看。")

def main():
    st.title("🤖 RoboCoin 数据集清洗与标注系统")
    vocab = load_vocabulary()
    
    # 侧边栏：全局路径配置
    with st.sidebar:
        st.header("📂 数据源配置")
        dataset_path = st.text_input("输入数据集根路径:", value="./data/hdf5")
        
        if st.button("1. 扫描与类型检查", type="primary"):
            if not os.path.exists(dataset_path):
                st.error("路径不存在！")
            else:
                inspector = DatasetInspector(dataset_path)
                inspector.scan()
                if inspector.check_consistency():
                    st.session_state['grouped_datasets'] = inspector.grouped_datasets
                    st.session_state['dataset_path'] = dataset_path
                    st.session_state['valid_paths'] = inspector.get_all_valid_paths()
                    st.success("扫描成功！")
                else:
                    st.error("一致性检查失败，请检查数据格式。")
        
        if 'valid_paths' in st.session_state:
            st.info(f"当前有效数据总量: {len(st.session_state['valid_paths'])} 条")

    # 主界面分为两个 Tab
    tab1, tab2 = st.tabs(["🔍 第一步：数据清洗与排查", "📝 第二步：元数据标注"])

    # ==========================================
    # TAB 1: 数据清洗与预览
    # ==========================================
    with tab1:
        if 'grouped_datasets' not in st.session_state:
            st.info("👈 请先在左侧输入路径并点击「扫描与类型检查」")
        else:
            grouped_datasets = st.session_state['grouped_datasets']
            valid_paths = st.session_state['valid_paths']
            target_dir = st.session_state['dataset_path']
            
            # --- 步骤 1: 处理混入的不同类型数据 ---
            st.subheader("1. 数据类型检查与整理")
            if len(grouped_datasets) > 1:
                st.warning(f"检测到 {len(grouped_datasets)} 种混合的数据类型: {list(grouped_datasets.keys())}")
                if st.button("自动分类并物理隔离不同类型数据"):
                    organizer = DatasetOrganizer(target_dir)
                    new_grouped_paths = organizer.sort_by_type(grouped_datasets, target_dir)
                    st.session_state['grouped_datasets'] = new_grouped_paths
                    st.success("✅ 数据已按照类型分组移动到独立文件夹中。")
                    st.rerun()
                
                selected_type = st.selectbox("请选择本次要处理的数据类型:", list(grouped_datasets.keys()))
                valid_paths = grouped_datasets[selected_type]
                st.session_state['valid_paths'] = valid_paths
            else:
                st.success(f"✅ 数据类型一致，仅包含: {list(grouped_datasets.keys())[0]}")

            # --- 步骤 2: 人工审核（剔除其他任务数据） ---
            st.subheader("2. 人工逐帧审核 (排查不同任务数据)")
            st.markdown("通过 Rerun 窗口检查是否混入了**无关任务/动作错误**的数据。")
            if st.button("🚀 启动人工审核 (Rerun)"):
                with st.spinner("请在弹出的 Rerun 窗口中操作 (使用键盘 N/P 切换, Space 标记异常, Esc 退出)..."):
                    viz = RerunVisualizer("RoboCoin_Review")
                    reviewer = DatasetReviewer(viz)
                    bad_datasets = reviewer.start_review(valid_paths)
                    
                    if bad_datasets:
                        organizer = DatasetOrganizer(target_dir)
                        quarantine_dir = organizer.quarantine_bad_data(bad_datasets, target_dir)
                        st.warning(f"🔒 发现异常任务数据！已将其隔离到: {quarantine_dir}")
                        final_paths = [p for p in valid_paths if p not in bad_datasets]
                        st.session_state['valid_paths'] = final_paths
                        st.success(f"🧹 剔除异常数据后，剩余有效数据: {len(final_paths)} 条")
                        st.rerun()  # 强制刷新状态
                    else:
                        st.success("✨ 完美！未发现混入的其他任务数据。")
                        st.rerun()

            # --- 步骤 3: 抽样对比预览 ---
            st.subheader("3. 抽样对比预览 (检查命名与视频内容)")
            st.markdown("自动抽取 首、中、尾 3个样本进行并排播放，以便宏观确认数据与标注预期是否相符。")
            if st.button("📺 启动多视角对比预览"):
                if len(valid_paths) == 0:
                    st.error("没有可用数据进行预览。")
                else:
                    indices = [0]
                    if len(valid_paths) > 1: indices.append(len(valid_paths)-1)
                    if len(valid_paths) > 2: indices.insert(1, len(valid_paths)//2)
                    sample_paths = [valid_paths[i] for i in indices]
                    run_parallel_preview(sample_paths)

    # ==========================================
    # TAB 2: 元数据标注 (生成 YAML) - 完全动态驱动版
    # ==========================================
    with tab2:
        st.header("任务元数据标注")
        
        # 1. 解析 JSON Schema，将字段按 Group 分组
        schema_fields = vocab.get("fields", [])
        groups = {}
        for f in schema_fields:
            g = f.get("group", "其他配置")
            if g not in groups: groups[g] = []
            groups[g].append(f)
            
        # 2. 动态收集表单数据
        collected_data = {}
        
        # 3. 双列布局渲染 (兼容预设分组与任意新增分组)
        col1, col2 = st.columns(2)
        
        # 预设在左列和右列的分组（保持原有布局的美观）
        col1_groups = ["基本信息", "场景设置"]
        col2_groups = ["动作与物品", "硬件配置", "其他配置"]
        
        # 自动发现 JSON 中用户新增的其他分组（例如你写的 "测试分组"）
        all_preset_groups = set(col1_groups + col2_groups)
        new_groups = [g for g in groups.keys() if g not in all_preset_groups]
        
        # 把新发现的分组均匀追加到左列和右列的后面
        for i, g in enumerate(new_groups):
            if i % 2 == 0:
                col1_groups.append(g)
            else:
                col2_groups.append(g)

        # 执行渲染
        with col1:
            for g in col1_groups:
                if g in groups:
                    st.subheader(g)
                    for field in groups[g]:
                        # 传入 collected_data，用于级联依赖的判断
                        collected_data[field["key"]] = render_field(field, collected_data)
                        
        with col2:
            for g in col2_groups:
                if g in groups:
                    st.subheader(g)
                    for field in groups[g]:
                        # 传入 collected_data，用于级联依赖的判断
                        collected_data[field["key"]] = render_field(field, collected_data)

        st.markdown("---")
        if st.button("💾 生成 YAML 标注文件", type="primary"):
            if 'dataset_path' not in st.session_state:
                st.warning("请先在「数据清洗与排查」页面加载数据！")
            elif not collected_data.get("dataset_name"):
                st.error("请填写数据集名称！")
            else:
                # 传入全自动收集到的字典
                save_path = ConfigGenerator.analyze_and_save(collected_data, st.session_state['dataset_path'])
                st.success(f"🎉 标注文件生成成功！\n文件路径: `{save_path}`")
                
                with st.expander("点击查看生成的 YAML 内容 (纯英文)"):
                    st.code(ConfigGenerator.generate_yaml_string(collected_data), language="yaml")

if __name__ == "__main__":
    main()