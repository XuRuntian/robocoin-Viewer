import struct

def quick_diagnose(file_path):
    print(f"--- 原始字节分析: {file_path} ---")
    try:
        with open(file_path, "rb") as f:
            # 1. 检查文件头
            header = f.read(8)
            print(f"文件头 (Hex): {header.hex()}")
            if b'MCAP' not in header:
                print("❌ 这可能不是一个标准的 MCAP 文件")
            
            # 2. 读取前 1MB 的内容，搜寻可疑字符串
            f.seek(0)
            chunk = f.read(1024 * 1024) # 读 1MB
            
            print("\n--- 关键字符串扫描 ---")
            targets = [b"topic", b"image", b"camera", b"sensor_msgs", b"foxglove", b"umi"]
            for t in targets:
                found = chunk.count(t)
                print(f"关键词 '{t.decode()}': 出现 {found} 次")

            # 3. 打印出前几个找到的 Topic 路径示例
            # 通过 '/' 符号寻找类似 /obs/images/top 的路径
            import re
            # 匹配类似 /a/b/c 格式的字符串
            paths = re.findall(rb'/[a-zA-Z0-9_/]+', chunk)
            unique_paths = sorted(list(set(paths)))[:10]
            print("\n--- 发现的疑似路径示例 ---")
            for p in unique_paths:
                try:
                    print(f"  {p.decode()}")
                except:
                    pass

    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    quick_diagnose("/home/user/test_data/mcap/20251223100047.mcap")