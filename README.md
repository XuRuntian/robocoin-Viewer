# RoboCoin Viewer 机器人数据查看器 🤖👁️

**基于 Rerun 的具身智能数据集自动化清洗、预览与管理工具**

## 🔍 核心亮点
- 📦 **多格式支持**：支持 Unitree、LeRobot、HDF5、ROS（bag/mcap）、原始文件夹等格式
- 🔍 **智能整理器**：自动识别混杂数据类型并分类归档
- 🕵️‍♂️ **交互式审阅**：基于 Rerun 的可视化界面 + 键盘快捷键操作
- 🎥 **并行预览**：首/中/尾帧并行比对快速质检
- 🧼 **隔离系统**：异常数据自动移动到带时间戳的隔离区

![RoboCoin Viewer 演示图](demo.gif) <!-- 图片占位符 -->

## 🛠️ 安装指南

### 环境要求
- Python `3.10`（参考 `.python-version`）
- [Rust](https://rust-lang.org)（Rerun 依赖）

### 安装步骤
```bash
# 克隆仓库
git clone git@github.com:XuRuntian/robocoin-Viewer.git

# 安装依赖
cd robocoin-Viewer
uv sync
```

## 🚀 快速入门
```bash
# 基础用法
uv run main.py /path/to/your/dataset

# 跳过交互审阅
uv run main.py /path/to/your/dataset --skip-review

# 跳过最终预览
uv run main.py /path/to/your/dataset --no-preview
```

### 键盘操作指南
| 按键       | 功能              |
|-----------|-------------------|
| `←`/`→`    | 切换上下数据集    |
| `B`       | 标记为异常样本    |
| `Esc`     | 退出审阅模式      |

## 📁 支持格式
| 格式       | 特征文件                  | 结构特点                     |
|-----------|----------------------------|------------------------------|
| Unitree   | `data.json`               | 人形机器人运动数据           |
| LeRobot   | `meta/info.json` 或 `*.parquet` | 支持智能 chunk 识别          |
| HDF5      | `*.hdf5`                  | 分层数据格式                 |
| ROS       | `*.bag` 或 `*.mcap`         | 支持 Bag/MCAP 文件格式       |
| 原始文件夹 | 任意未结构化数据的文件夹     | 基础文件整理功能             |

## 🌟 工作流程
### 步骤 1：扫描与整理
- 递归扫描目录自动识别混合数据类型
- 自动生成类型专属归档目录：
  - `Unitree_Data/`
  - `HDF5_Data/`
  - `_QUARANTINE_YYYYMMDD/`（异常数据区）
- 智能 chunk 识别防止误操作

### 步骤 2：交互式审阅
- 在 Rerun 查看器中可视化序列
- 键盘快捷键快速导航
- 使用 `B` 键标记异常样本

### 步骤 3：数据隔离
- 异常样本自动移动到带时间戳的隔离目录
- 生成包含以下信息的清单文件：
  - 原始路径
  - 隔离时间戳
  - 隔离原因

## 🧩 项目结构
```
src/
├── core/               # 核心逻辑与业务规则
├── adapters/           # 格式适配器模块
└── ui/                 # Rerun 可视化组件
```

## 📆 未来规划
- 🛠️ 生成配置文件的常用工作流
- 📊 增强分析仪表盘
- 📤 清洗后数据集的导出功能