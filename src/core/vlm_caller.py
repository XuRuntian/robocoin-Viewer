import os
import json
import base64
import re
from openai import OpenAI
import httpx # 用于配置底层网络超时和代理

def encode_image_to_base64(image_path):
    """将本地图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json_from_response(text: str) -> list:
    """健壮的 JSON 提取器"""
    json_pattern = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)
    match = json_pattern.search(text)
    if match:
        json_str = match.group(1)
    else:
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
        else:
            json_str = text
            
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法从 VLM 响应中解析 JSON。原始响应:\n{text}\n错误: {e}")

def call_qwen_vl_api(image_path: str, global_task_desc: str) -> list:
    """调用阿里云 Qwen-VL 模型并返回解析好的 JSON List"""
    
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请先设置！")

    # 1. 强制阿里云域名不走系统 VPN/代理（国内服务器挂代理必断连）
    os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,aliyuncs.com"

    # 2. 修复了 URL 格式，并配置底层的 http 客户端（设置 120 秒超时）
    http_client = httpx.Client(
        timeout=httpx.Timeout(120.0), # 给大图片上传留足时间
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 👈 这里的网址必须是纯净的！
        http_client=http_client
    )

    prompt_text = f"""You are an expert robotic data annotator. You are given a 3x3 image grid containing 9 sequential keyframes (labeled [1] to [9]) of a dual-arm robot performing a manipulation task.

[Global Context & Prior Knowledge]
The user has provided the following background information about the task:
"{global_task_desc}"
CRITICAL: Strictly adhere to this context. Do not hallucinate objects that are not mentioned or clearly visible. Identify the objects exactly as described above.

[Granularity & Action Rules]
You must segment the entire process into logical "Subtasks". A proper subtask MUST follow these rules:
1. Identify the actor: Explicitly start with "Left hand", "Right hand", or "Both hands".
2. Use standard primitive verbs: "approaches", "grasps", "lifts", "moves", "places", "releases".
3. "Approach and Grasp" is usually grouped into ONE subtask. "Lift, Move, and Place" is usually grouped into ONE subtask.
4. Handover between hands MUST be clearly separated.

[Language Requirement]
Even if the Global Context is provided in Chinese, the final `instruction` values in the JSON MUST BE TRANSLATED TO AND WRITTEN IN ENGLISH.

[Output Format]
Output ONLY a strict JSON list of subtasks. Do not output any reasoning, thinking process, or explanatory text.
The JSON must strictly contain these keys: "subtask_id" (int), "instruction" (English string), "start_image" (int, 1-9), "end_image" (int, 1-9).
"""

    base64_image = encode_image_to_base64(image_path)
    print(f"📸 图片转 Base64 成功，体积大小约为: {len(base64_image) / 1024 / 1024:.2f} MB")
    print(f"🚀 正在建立与阿里云百炼的连接，请耐心等待 10-30 秒...")

    try:
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1, 
        )
    except Exception as e:
        print(f"❌ 网络请求阶段抛出异常: {e}")
        raise e

    print(f"✅ 已收到 VLM 模型的响应，正在解析...")
    raw_output = response.choices[0].message.content
    print(f"\n[VLM 原始返回]\n{raw_output}\n")
    
    return extract_json_from_response(raw_output)