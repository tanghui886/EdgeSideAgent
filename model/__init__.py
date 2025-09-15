from langchain_openai import ChatOpenAI

# 不同模型对应的端口配置
MODEL_CONFIGS={
    "Qwen2.5-VL-7B-Instruct-lora":{
        "model_name":"model/Qwen2.5-VL-7B-Instruct-lora",
        "base_url":"http://127.0.0.1:8001/v1",
        "api_key":"xxxxxx"
    },
    "Qwen2.5-VL-7B-Instruct":{
        "model_name":"model/Qwen2.5-VL-7B-Instruct",
        "base_url":"http://127.0.0.1:8002/v1",
        "api_key":"xxxxxx"
    }
}

# 可用模型列表
AVAILABLE_MODELS=list(MODEL_CONFIGS.keys())


def create_multiModal_llm(model_name=None):
    if model_name is None:
        model_name=AVAILABLE_MODELS[0]
    """创建多模态大模型实例"""
    config=MODEL_CONFIGS.get(model_name)

    return ChatOpenAI(
        model=config["model_name"],
        api_key=config["api_key"],
        base_url=config["base_url"],
    )


# 导出所有需要的变量
__all__=['create_multiModal_llm','AVAILABLE_MODELS','MODEL_CONFIGS']