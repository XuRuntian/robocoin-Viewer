# src/core/reviewer.py
import time
import threading
from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb
from pynput import keyboard  # 引入键盘监听库
from src.core.factory import ReaderFactory

class DatasetReviewer:
    def __init__(self, visualizer):
        """
        :param visualizer: RerunVisualizer 实例
        """
        self.viz = visualizer
        self.bad_datasets = [] 
        self.current_idx = 0
        self.dataset_paths = []
        self.is_running = False
        self.needs_refresh = False # 标记是否需要刷新画面

    def start_review(self, dataset_paths: list):
        """
        启动交互式审核流程 (全局键盘监听版)
        """
        if not dataset_paths:
            print("❌ 没有数据可审核")
            return []

        self.dataset_paths = dataset_paths
        self.current_idx = 0
        self.is_running = True
        self.needs_refresh = True # 初始刷新

        print("\n" + "="*50)
        print("🕵️‍♂️ 进入交互式审核模式 (键盘控制)")
        print("="*50)
        print("保持 Rerun 窗口在前台即可，使用以下按键控制：")
        print("  [→] 右箭头 / N : 下一个数据")
        print("  [←] 左箭头 / P : 上一个数据")
        print("  B / b    : 标记/取消标记【异常】")
        print("  [Esc] / Q      : 退出审核")
        print("-" * 50)

        # 1. 设置布局
        self._setup_review_layout()

        # 2. 启动键盘监听 (非阻塞)
        listener = keyboard.Listener(on_release=self._on_key_release)
        listener.start()

        # 3. 主循环 (代替原来的 input 阻塞)
        try:
            while self.is_running:
                if self.needs_refresh:
                    self._refresh_view()
                    self.needs_refresh = False
                
                # 降低 CPU 占用，轮询频率 10Hz 足够了
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()

        return self.bad_datasets

    def _on_key_release(self, key):
        """
        键盘回调函数 (运行在独立线程中)
        """
        if not self.is_running:
            return False # 停止监听

        try:
            # 获取键值
            k = None
            if hasattr(key, 'char'):
                k = key.char  # 字母键
            else:
                k = key       # 特殊键 (Key.right, Key.esc)

            # 逻辑映射
            if k == 'q' or k == keyboard.Key.esc:
                print("\n⏹ 退出审核...")
                self.is_running = False
                return False
            
            elif k == 'n' or k == keyboard.Key.right:
                if self.current_idx < len(self.dataset_paths) - 1:
                    self.current_idx += 1
                    self.needs_refresh = True
                else:
                    print("\n🎉 已经是最后一个了！(按 Q 退出)")

            elif k == 'p' or k == keyboard.Key.left:
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.needs_refresh = True
                else:
                    print("\n已经是第一个了！")

            elif k == 'b':
                self._toggle_bad_mark()
                self.needs_refresh = True

        except Exception as e:
            print(f"Key Error: {e}")

    def _toggle_bad_mark(self):
        path = self.dataset_paths[self.current_idx]
        name = Path(path).name
        if path not in self.bad_datasets:
            self.bad_datasets.append(path)
            print(f"⚠️ [标记异常]: {name}")
        else:
            self.bad_datasets.remove(path)
            print(f"👌 [取消标记]: {name}")

    def _refresh_view(self):
        """
        刷新当前数据的画面
        """
        path = self.dataset_paths[self.current_idx]
        name = Path(path).name
        
        # 终端打印进度 (覆盖式打印，防止刷屏)
        status_icon = "❌ BAD" if path in self.bad_datasets else "✅ OK"
        print(f"\r[{self.current_idx+1}/{len(self.dataset_paths)}] 审核中: {name} | 状态: {status_icon}    ", end="", flush=True)

        self._show_dataset_snapshot(path)

    def _setup_review_layout(self):
        """
        设置 Rerun 布局：强制显示 3 列
        """
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.TextDocumentView(origin="review/info", name="Dataset Info"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="review/0_start", name="Start (0%)"),
                    rrb.Spatial2DView(origin="review/1_mid", name="Mid (50%)"),
                    rrb.Spatial2DView(origin="review/2_end", name="End (100%)"),
                ),
                row_shares=[1, 4]
            ),
            collapse_panels=True
        )
        rr.send_blueprint(blueprint)

    def _show_dataset_snapshot(self, path: str):
        """
        提取 3 帧并显示
        """
        rr.log("review", rr.Clear(recursive=True))

        try:
            reader = ReaderFactory.get_reader(path)
            if not reader.load(path):
                rr.log("review/info", rr.TextDocument(f"❌ Load Failed: {Path(path).name}"))
                return

            length = reader.get_length()
            if length == 0:
                rr.log("review/info", rr.TextDocument(f"⚠️ Empty Dataset"))
                return

            indices = {
                "0_start": 0,
                "1_mid": length // 2,
                "2_end": length - 1
            }

            # 动态更新 Info 面板
            status_text = "🔴 **BAD DATA**" if path in self.bad_datasets else "🟢 **GOOD DATA**"
            info_text = f"# {Path(path).name}\n\n"
            info_text += f"**Frames**: {length}\n"
            info_text += f"**Type**: {type(reader).__name__}\n"
            info_text += f"**Status**: {status_text}\n"
            info_text += "\n---\n**Controls**:\n[→] Next | [←] Prev | [Space] Mark Bad | [Esc] Quit"
            
            rr.log("review/info", rr.TextDocument(info_text, media_type="text/markdown"))

            for prefix, idx in indices.items():
                frame = reader.get_frame(idx)
                for cam_name, img in frame.images.items():
                    rr.log(f"review/{prefix}/{cam_name}", rr.Image(img))
            
            reader.close()

        except Exception as e:
            # print(f"\nError reviewing {path}: {e}")
            pass