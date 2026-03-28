import os
import time
from datetime import datetime
from typing import Any

import dashscope
import httpx
from dashscope import VideoSynthesis
from http import HTTPStatus


OUTPUT_DIR = r"D:\Project_Code\Qwen\Video\Output"


def validate_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def require_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        api_key = input("未检测到 DASHSCOPE_API_KEY，请输入 API Key: ").strip()
        if not api_key:
            raise RuntimeError("未提供有效 API Key。")
    return api_key


def submit_video_task(call_kwargs: dict[str, Any]):
    model_name = str(call_kwargs.get("model", ""))
    has_ref_video = bool(call_kwargs.get("reference_video_urls"))
    has_img_url = bool(call_kwargs.get("img_url"))
    if "r2v" in model_name and not has_ref_video:
        print("警告: 当前模型是 r2v，通常需要 reference_video_urls，缺失时任务可能 FAILED。")
    if "i2v" in model_name and not has_img_url:
        print("警告: 当前模型是 i2v，必须传入 img_url，否则任务会 FAILED。")

    response = VideoSynthesis.async_call(**call_kwargs)
    if response.status_code == HTTPStatus.OK:
        return response

    if response.status_code == HTTPStatus.UNAUTHORIZED and response.code == "InvalidApiKey":
        print("当前 API Key 无效，请重新输入后重试。")
        new_key = input("请输入新的 DashScope API Key: ").strip()
        if not new_key:
            raise RuntimeError("API Key 为空，任务已终止。")

        dashscope.api_key = new_key
        response = VideoSynthesis.async_call(**call_kwargs)
        if response.status_code == HTTPStatus.OK:
            return response

    raise RuntimeError(
        f"Task creation failed: status={response.status_code}, code={response.code}, message={response.message}"
    )


def download_video(video_url: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"wan26_r2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(output_dir, file_name)

    with httpx.stream("GET", video_url, timeout=300.0) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return output_path


def wait_for_task_with_progress(task_id: str, interval_seconds: int = 3, max_wait_seconds: int = 900):
    start = time.time()
    attempt = 0

    while True:
        attempt += 1
        elapsed = int(time.time() - start)
        response = VideoSynthesis.fetch(task_id)

        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"Task status fetch failed: status={response.status_code}, code={response.code}, message={response.message}"
            )

        status = response.output.task_status
        print(f"[{attempt}] {elapsed}s -> task_status={status}")

        if status in {"SUCCEEDED", "FAILED", "CANCELED"}:
            return response

        if elapsed >= max_wait_seconds:
            raise RuntimeError(f"等待超时: {elapsed}s，任务仍未完成，task_id={task_id}")

        time.sleep(interval_seconds)


def get_reference_image_url() -> str | None:
    """Read an optional reference image URL from terminal input."""
    default_url = "https://d.tmpfile.link/public/2026-03-28/409668dc-d166-4082-9832-508ba115dc6c/e131bc2b82504e23971cd4ce3cae2bae.png"
    value = input(f"请输入参考图 URL（可选，回车使用默认值）[{default_url}]: ").strip()
    if not value:
        value = default_url

    if not validate_http_url(value):
        raise ValueError("参考图 URL 必须以 http:// 或 https:// 开头")

    return value


def main() -> None:
    dashscope.api_key = require_api_key()

    prompt = "少女原地跳宅舞，轻快踏步，双手左右摆动，身体轻微左右晃动，头发和裙摆自然飘动，动作流畅柔和，节奏轻快，全身动态，表情甜美活泼，镜头轻微跟随，无多余晃动，画质清晰，动态自然不扭曲"
    # reference_video_url = "https://cdn.wanx.aliyuncs.com/static/demo-wan26/vace.mp4"
    reference_image_url = get_reference_image_url()

    call_kwargs = {
        "model": "wan2.6-i2v-flash",
        "prompt": prompt,
        # "reference_video_urls": [reference_video_url],
        "size": "1280*720",
        "duration": 15,
        "shot_type": "multi",
    }
    if reference_image_url:
        call_kwargs["img_url"] = reference_image_url

    response = submit_video_task(call_kwargs)

    task_id = response.output.task_id
    print(f"Task created: {task_id}")

    final_response = wait_for_task_with_progress(task_id, interval_seconds=3)
    if final_response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Task failed: status={final_response.status_code}, code={final_response.code}, message={final_response.message}"
        )

    task_status = final_response.output.task_status
    if task_status != "SUCCEEDED":
        output_dict = {}
        if final_response.output is not None:
            try:
                output_dict = dict(final_response.output)
            except Exception:
                output_dict = {"raw_output": str(final_response.output)}

        raise RuntimeError(
            "Task ended with status: "
            f"{task_status}, code={final_response.code}, message={final_response.message}, output={output_dict}"
        )

    video_url = final_response.output.video_url
    if not video_url:
        raise RuntimeError("Task succeeded but no video_url was returned.")

    local_path = download_video(video_url, OUTPUT_DIR)
    print(f"Video URL: {video_url}")
    print(f"Saved to: {local_path}")


if __name__ == "__main__":
    main()