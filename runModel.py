import torch
from PIL import Image
import base64
from io import BytesIO
import uvicorn
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List,Optional,Union
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
import time
import uuid
import os
import argparse

# ========================
#python runModel.py --modelPath /home/modelscope/ai/model/Qwen2.5-VL-7B-Instruct --modelName Qwen2.5-VL-7B-Instruct --port 8002
# ✅ 解析命令行参数
parser = argparse.ArgumentParser(description="Model API Server")
parser.add_argument("--modelPath", type=str, required=True, help="modelPath to the model directory")
parser.add_argument("--modelName", type=str, required=True, help="modelName to the model name")
parser.add_argument("--port", type=int, default=8001, help="Port to run the server on (default: 8001)")
parser.add_argument("--dtype", type=str, default=torch.bfloat16, help="torch_dtype (default: torch.bfloat16)")
args = parser.parse_args()
# ========================
model_path=args.modelPath
model_name=args.modelName
port=args.port
dtype=args.dtype
print(f"🧠 正在加载 {model_name}（Transformers + ROCm GPU）...")
# 检测设备
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  使用设备: {device}")

try:
    from transformers import AutoModelForCausalLM
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="eager",  # 使用eager模式，兼容性更好 flash_attention_2 flash_attention_3
        torch_dtype=dtype,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True,
    )

    processor=AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print("✅ 模型加载完成！")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    try:
        model=AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_4bit=False,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            model=model.to(device)

        processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
        print("✅ 模型使用备用方式加载完成！")
    except Exception as e2:
        print(f"❌ 备用加载方式也失败: {e2}")
        raise RuntimeError("无法加载模型")

# ========================
# 🌐 FastAPI 初始化
# ========================
app=FastAPI(title="Qwen2.5-VL API Server (ROCm Native)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
# 📥 请求体定义（兼容 OpenAI 格式）
# ========================
class ImageUrlContent(BaseModel):
    url: str


class ImageUrlPart(BaseModel):
    type: str="image_url"
    image_url: ImageUrlContent


class TextPart(BaseModel):
    type: str="text"
    text: str


ContentPart=Union[TextPart,ImageUrlPart]


class Message(BaseModel):
    role: str
    content: Union[str,List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str=model_name
    messages: List[Message]
    max_tokens: Optional[int]=512
    temperature: Optional[float]=0.7
    stream: Optional[bool]=False


# ========================
# 📤 响应体定义
# ========================
class ChatMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int=0
    message: ChatMessage
    finish_reason: str="stop"


class Usage(BaseModel):
    prompt_tokens: int=0
    completion_tokens: int=0
    total_tokens: int=0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str="chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


# ========================
# 🖼️ 工具函数：Base64 → PIL Image
# ========================
def base64_to_image(b64_str: str) -> Image.Image:
    try:
        if b64_str.startswith("data:image"):
            b64_str=b64_str.split(",")[1]
        image_data=base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400,detail=f"Invalid image base64: {str(e)}")


# ========================
# 🚀 核心推理函数
# ========================
def generate_response(messages: List[Message], max_tokens: int, temperature: float):
    try:
        # 构建消息内容
        text_content = ""
        images = []

        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                text_content += content + "\n"
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, TextPart):
                        text_content += part.text + "\n"
                    elif isinstance(part, ImageUrlPart):
                        img = base64_to_image(part.image_url.url)
                        images.append(img)

        # 准备模型输入
        messages_for_model = [
            {
                "role": "user",
                "content": []
            }
        ]

        # 添加图像
        if images:
            temp_image_path = f"temp_image_{uuid.uuid4().hex}.jpg"
            images[0].save(temp_image_path, quality=95, optimize=True)
            messages_for_model[0]["content"].append(
                {"type": "image", "image": temp_image_path}
            )

        # 添加文本
        messages_for_model[0]["content"].append(
            {"type": "text", "text": text_content.strip()}
        )

        # 使用processor应用聊天模板
        text = processor.apply_chat_template(
            messages_for_model,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理视觉信息
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages_for_model)
        except ImportError:
            # 如果没有 qwen_vl_utils，使用备用方式
            image_inputs = [images[0]] if images else None
            video_inputs = None

        # 准备输入
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        input_ids = inputs.input_ids  # 保存输入 token ids

        # 生成响应
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )

        # 裁剪掉输入部分，只保留生成部分
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 清理临时文件
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # ✅ 返回三元组：文本、输入token数、输出token数
        prompt_token_count = input_ids.shape[1]  # 输入序列长度
        completion_token_count = generated_ids_trimmed[0].shape[0]  # 输出序列长度

        return output_text, prompt_token_count, completion_token_count

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ========================
# 🌐 API 端点：/v1/chat/completions
# ========================
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet.")

    start_time = time.time()

    # ✅ 现在 generate_response 返回三个值
    response_text, prompt_tokens, completion_tokens = generate_response(
        request.messages, request.max_tokens, request.temperature
    )

    end_time = time.time()
    total_time = end_time - start_time
    tps = completion_tokens / total_time if total_time > 0 else 0

    # 🖨️ 打印性能统计
    print(f"\n📊 [性能统计]")
    print(f"📥 输入 tokens: {prompt_tokens}")
    print(f"📤 输出 tokens: {completion_tokens}")
    print(f"⏱️  生成速度: {tps:.2f} tokens/s")
    print(f"⏳ 总耗时: {total_time:.3f} 秒\n")

    total_tokens = prompt_tokens + completion_tokens

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=response_text)
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )


# 健康检查端点
@app.get("/health")
async def health_check():
    return {
        "status":"healthy",
        "device":str(model.device),
        "dtype":str(model.dtype),
        "model":model_name
    }


# ========================
# 🏁 启动服务
# ========================
if __name__ == "__main__":
    print(f"🚀 {model_name} API Server 启动中...")
    print(f"🖥️  设备: {device}")
    print(f"📊 精度: {model.dtype}")
    print(f"🌐 交互文档: http://localhost:{port}/docs")
    print(f"📌 请求地址: POST http://localhost:{port}/v1/chat/completions")
    print(f"🏥 健康检查: GET http://localhost:{port}/health")

    uvicorn.run(app,host="0.0.0.0",port=port)