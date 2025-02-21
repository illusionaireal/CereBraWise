from dataclasses import dataclass
from typing import Optional, List, Dict, Union
import base64
from PIL import Image
from openai import OpenAI
import io
from langchain.schema.runnable import RunnableLambda
import os


@dataclass
class ProcessorOutput:
    """处理器输出数据类"""
    vector: Optional[List[float]] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class ClientManager:
    """API客户端管理器（单例模式）"""
    _instance = None
    _client = None
    _api_key = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self, api_key: str = None) -> Optional[OpenAI]:
        """获取API客户端，如果配置变化则重新创建"""
        # 加载最新的环境变量
        current_api_key = api_key or os.getenv("NVIDIA_API_KEY")
            
        # 如果配置变化，重新创建客户端
        if (current_api_key != self._api_key):
            try:
                self._client = OpenAI(
                    api_key=current_api_key,
                    base_url="https://integrate.api.nvidia.com/v1"
                )
                self._api_key = current_api_key
            except Exception as e:
                print(f"初始化API客户端失败: {e}")
                self._reset()
                
        return self._client
    
    def _reset(self):
        """重置客户端状态"""
        self._client = None
        self._api_key = None

class Preprocessor:
    def __init__(self, api_key: str = None):
        """初始化预处理器"""
        self.process_chain = RunnableLambda(self._process)
        self.client_manager = ClientManager()
        self.api_key = api_key
        self.preprocess_chain = RunnableLambda(self._preprocess)        
        
    def _process(self, inputs: Dict[str, str]) -> ProcessorOutput:
        """
        RunnableLambda处理函数
        
        Args:
            inputs: {
                "image": Union[str, Image.Image],
                "prompt": str = "tourist attraction"
            }
            
        Returns:
            ProcessorOutput: 处理结果
        """
        try:
            # 检查输入
            image_b64 = inputs.get("image_b64")
            if not image_b64:
                return ProcessorOutput(error="未提供图片")
                
            # 获取客户端
            client = self.client_manager.get_client(self.api_key)
            if not client:
                return ProcessorOutput(error="未配置API Key或初始化失败")
            
            prompt = inputs.get("prompt", "tourist attraction")
            
            # 使用NVIDIA CLIP生成向量
            response = client.embeddings.create(
                input=[
                    prompt,
                    f"data:image/png;base64,{image_b64}"
                ],
                model=inputs.get("model", "nvidia/nvclip"),
                encoding_format="float"
            )
            
            return ProcessorOutput(
                vector=response.data[0].embedding,
                metadata={
                    "prompt": prompt,
                    "model": inputs.get("model", "nvidia/nvclip")
                }
            )
            
        except Exception as e:
            return ProcessorOutput(
                error=f"处理错误: {str(e)}"
            )
        
    def preprocess(
        self,
        image_b64: str,
        prompt: str = "tourist attraction",
        model: str = "nvidia/nvclip"    
    ) -> ProcessorOutput:
        return self._process({
            "image_b64": image_b64,
            "prompt": prompt,
            "model": model
        })
    
    def _preprocess(self, image_b64: str, prompt: str = "tourist attraction", model: str = "nvidia/nvclip") -> ProcessorOutput:
        return self.process_chain.invoke({
            "image_b64": image_b64,
            "prompt": prompt,
            "model": model
        })

def convert_image_to_base64(image_path: str) -> str:
    """将图片转换为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 使用示例
def main():
    preprocessor = Preprocessor(api_key="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")

    convert_image_to_base64_chain = RunnableLambda(convert_image_to_base64)

    
    # 处理图片并直接获取向量
    result = (convert_image_to_base64_chain | preprocessor.preprocess_chain).invoke("nv_preprocess/images/tiananmen.png")
    
    if result.vector:
        print(f"向量生成成功，维度: {len(result.vector)}")
    

if __name__ == "__main__":
    main()