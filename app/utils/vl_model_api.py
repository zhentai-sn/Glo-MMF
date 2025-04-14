from openai import OpenAI

def init_openai_client(base_url: str = 'http://127.0.0.1:8000/v1', api_key: str = 'EMPTY') -> OpenAI:
    """初始化 OpenAI 客户端

    Args:
        base_url (str, optional): API基础URL. Defaults to 'http://127.0.0.1:8000/v1'.
        api_key (str, optional): API密钥. Defaults to 'EMPTY'.

    Returns:
        OpenAI: OpenAI客户端实例
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client

def query_model(client: OpenAI, messages, max_tokens: int = 2048, temperature: float = 0) -> str:
    """查询模型获取响应

    Args:
        client (OpenAI): OpenAI客户端实例
        query (str): 查询文本
        max_tokens (int, optional): 最大生成token数. Defaults to 512.
        temperature (float, optional): 采样温度. Defaults to 0.

    Returns:
        str: 模型响应文本
    """
    models = [model.id for model in client.models.list().data]
    if not models:
        raise ValueError("No available models found")
    
    # messages = [{'role': 'user', 'content': query}]
    resp = client.chat.completions.create(
        model=models[0], 
        messages=messages, 
        max_tokens=max_tokens, 
        temperature=temperature
    )
    return resp.choices[0].message.content

def query_model_with_image(client: OpenAI, image_path: str, text: str, max_tokens: int = 2048, temperature: float = 0) -> str:
    """带图像的模型查询

    Args:
        client (OpenAI): OpenAI客户端实例
        image_path (str): 图像文件路径
        text (str): 查询文本
        max_tokens (int, optional): 最大生成token数. Defaults to 2048.
        temperature (float, optional): 采样温度. Defaults to 0.

    Returns:
        str: 模型响应文本
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": text}
        ]
    }]
    
    return query_model(client, messages, max_tokens, temperature)

# 使用示例更新
if __name__ == "__main__":
    client = init_openai_client()
    # image_path = "/public/zhangzhentai/code/glo_dmu/app/frontend/images/examples/test-151441.jpg"
    image_path = "/public/zhangzhentai/code/glo_dmu/app/frontend/images/examples/test-46002.jpg"

    # text = "请观察这张肾活检透射电镜图像，推测这位病人的毛细血管、基底膜、足突、电子致密物沉积、系膜区、肾小管和肾间质的状态"
    # text = "请观察这张肾活检透射电镜图像，描述的毛细血管、基底膜、足突、电子致密物沉积、系膜区、肾小管和肾间质的状态"
    text = "请观察这张肾活检透射电镜图像，描述的毛细血管、基底膜、足突、电子致密物沉积、系膜区的状态"
    # text = "请观察这张肾活检透射电镜图像，描述毛细血管、基底膜、足突、电子致密物沉积、系膜区的状态。病理专家现在告知：基底膜未明显增厚，厚度约300-350nm;足突节段性融合;系膜区可见电子致密物沉积"
    response = query_model_with_image(client, image_path, text)
    print(f'Query: {text}')
    print(f'Response: {response}')