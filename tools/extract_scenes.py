import pandas as pd
import json

def extract_excel_to_schema(excel_path):
    # 1. 读取 Excel 文件
    # header=None 表示我们通过列索引(0, 1, 2, 3)来定位数据
    # 如果你的表格第一行是表头（比如写着"一级场景（中文）"），你可以把 skiprows=1 加上
    df = pd.read_excel(excel_path, header=None)
    
    # 假设列的顺序严格按照你的描述：
    # 第0列: 一级中文, 第1列: 一级英文, 第2列: 二级中文, 第3列: 二级英文
    
    # 2. 处理合并单元格：向下填充 (ffill)
    df[0] = df[0].ffill()
    df[1] = df[1].ffill()
    
    # 清理一下可能存在的空白字符和 NaN 数据
    df = df.dropna(subset=[2, 3]) # 如果二级场景为空则跳过
    
    level1_options = {}
    level2_options_map = {}
    
    # 3. 遍历每一行，构建字典
    for index, row in df.iterrows():
        # 转为字符串并去除两端空格
        l1_cn = str(row[0]).strip()
        l1_en = str(row[1]).strip()
        l2_cn = str(row[2]).strip()
        l2_en = str(row[3]).strip()
        
        # 过滤掉可能的表头所在行（如果第一行是表头的话）
        if "一级场景" in l1_cn:
            continue
            
        # 组装一级场景
        if l1_en not in level1_options:
            level1_options[l1_en] = l1_cn
            level2_options_map[l1_en] = {}
            
        # 组装二级场景
        level2_options_map[l1_en][l2_en] = l2_cn
        
    # 4. 生成最终的 Schema 格式
    schema_output = [
        {
            "key": "scene_level1",
            "label": "一级场景",
            "type": "selectbox",
            "options": level1_options,
            "group": "场景设置"
        },
        {
            "key": "scene_level2",
            "label": "二级场景",
            "type": "selectbox_dependent",
            "depends_on": "scene_level1",
            "options_map": level2_options_map,
            "group": "场景设置"
        }
    ]
    
    # 打印出格式化好的 JSON 字符串
    print("👇 请将以下内容复制并替换到 vocabulary.json 的 fields 列表中对应位置：\n")
    print(json.dumps(schema_output, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    # 这里换成你实际的 excel 文件名
    excel_file = "/home/user/下载/场景树.xlsx" 
    extract_excel_to_schema(excel_file)