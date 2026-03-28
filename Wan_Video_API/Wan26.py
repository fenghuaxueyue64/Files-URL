import dashscope
import os
import time
import httpx
from http import HTTPStatus
from datetime import datetime

# 设置 API Key
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# ============ 在终端中输入 URL ============
print("=" * 60)
print("📋 请输入参考素材的公开访问 URL")
print("=" * 60)
image_url = input("请输入图片 URL: ").strip()
video_url = input("请输入视频 URL: ").strip()

# 验证 URL 格式
if not image_url.startswith('http://') and not image_url.startswith('https://'):
    print("❌ 图片 URL 必须是 http:// 或 https:// 开头")
    exit()
if not video_url.startswith('http://') and not video_url.startswith('https://'):
    print("❌ 视频 URL 必须是 http:// 或 https:// 开头")
    exit()

print(f"\n✅ 图片 URL: {image_url}")
print(f"✅ 视频 URL: {video_url}")

# ============ 调用视频生成 API ============
print("\n" + "=" * 60)
print("📤 提交视频生成任务")
print("=" * 60)

synthesis_url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis'
headers = {
    'X-DashScope-Async': 'enable',
    'Authorization': f'Bearer {dashscope.api_key}',
    'Content-Type': 'application/json'
}

data = {
    "model": "wan2.6-r2v-flash",
    "input": {
        "prompt": "美丽的少女穿着空气上衣，站在海边，微风吹动她的头发和衣服，阳光洒在她的脸上，展现出青春活力和自然美感。画面充满了夏日的氛围和浪漫的感觉。",
        "reference_urls": [image_url, video_url]
    },
    "parameters": {
        "size": "1920*1080",
        "duration": 10,
        "audio": True,
        "shot_type": "multi",
        "watermark": False,
        "prompt_extend": True,
    }
}

with httpx.Client(timeout=60.0) as client:
    # 提交任务
    response = client.post(synthesis_url, headers=headers, json=data)
    result = response.json()
    
    if response.status_code != 200:
        print(f"❌ 任务提交失败: {response.status_code}")
        print(f"错误信息: {result}")
        exit()
    
    task_id = result.get('output', {}).get('task_id')
    if not task_id:
        print("❌ 未获取到 Task ID")
        print(f"响应内容: {result}")
        exit()
    
    print(f"✅ 任务提交成功，Task ID: {task_id}")
    print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ============ 轮询任务状态（每 2 秒） ============
    print("\n⏳ 等待视频生成中...")
    print("-" * 60)
    
    query_url = f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
    query_headers = {
        'Authorization': f'Bearer {dashscope.api_key}',
        'Content-Type': 'application/json'
    }
    
    start_time = time.time()
    max_wait_time = 600  # 最多等待 10 分钟
    attempt = 0
    
    while True:
        time.sleep(2)  # 每 2 秒查询一次
        attempt += 1
        elapsed_time = time.time() - start_time
        
        try:
            query_response = client.get(query_url, headers=query_headers)
            query_result = query_response.json()
            
            status = query_result.get('output', {}).get('task_status', 'UNKNOWN')
            elapsed_str = f"{elapsed_time:.1f}秒"
            
            # 状态显示
            if status == 'RUNNING':
                print(f"   [{attempt}] 🕐 {elapsed_str} - 状态: RUNNING (生成中...)")
            elif status == 'SUCCEEDED':
                video_url_result = query_result.get('output', {}).get('video_url', '')
                print(f"   [{attempt}] ✅ {elapsed_str} - 状态: SUCCEEDED")
                print("\n" + "=" * 60)
                print("🎉 视频生成成功!")
                print("=" * 60)
                print(f"📹 视频 URL: {video_url_result}")
                print(f"⏱️  总耗时: {elapsed_str}")
                print(f"🕐 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                break
            elif status in ['FAILED', 'CANCELED']:
                message = query_result.get('output', {}).get('message', '未知错误')
                print(f"   [{attempt}] ❌ {elapsed_str} - 状态: {status}")
                print("\n" + "=" * 60)
                print("❌ 任务失败")
                print("=" * 60)
                print(f"错误信息: {message}")
                break
            elif status == 'PENDING':
                print(f"   [{attempt}] ⏳ {elapsed_str} - 状态: PENDING (排队中...)")
            else:
                print(f"   [{attempt}] ❓ {elapsed_str} - 状态: {status}")
            
            # 超时检查
            if elapsed_time > max_wait_time:
                print("\n" + "=" * 60)
                print("⚠️ 等待超时")
                print("=" * 60)
                print(f"已等待 {elapsed_str}，超过最大限制 {max_wait_time}秒")
                print(f"Task ID: {task_id}")
                print("您可以稍后手动查询任务状态")
                break
                
        except Exception as e:
            print(f"   [{attempt}] ⚠️ 查询失败: {e}")
            continue