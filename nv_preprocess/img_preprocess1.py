from dataclasses import dataclass
from typing import Optional, List, Dict
import base64
from langchain.schema.runnable import RunnableLambda
import os
import uuid 

import requests
os.environ["NVIDIA_API_KEY"] = "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"

header_auth = f"Bearer {os.getenv('NVIDIA_API_KEY')}"

def _upload_asset(input, description):
    """
    Uploads an asset to the NVCF API.
    :param input: The binary asset to upload
    :param description: A description of the asset

    """
    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

    headers = {
        "Authorization": header_auth,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "image/jpeg",
    }

    payload = {"contentType": "image/jpeg", "description": description}

    response = requests.post(assets_url, headers=headers, json=payload, timeout=30)

    response.raise_for_status()

    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    response = requests.put(
        asset_url,
        data=input,
        headers=s3_headers,
        timeout=300,
    )

    response.raise_for_status()
    return uuid.UUID(asset_id)


@dataclass
class ProcessorOutput:
    """处理器输出数据类"""
    vector: Optional[List[float]] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None


class Preprocessor:
    def __init__(self):
        """初始化预处理器"""
        self.process_chain = RunnableLambda(self._process)
        self.nvai_url="https://ai.api.nvidia.com/v1/cv/nvidia/nv-dinov2"
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
                
            # For images of size less than 200 KB send as base64 string
            payload = {
            "messages": [
                {
                "content": {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
                }
            ],
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": header_auth,
                "Accept": "application/json"
            }

            response = requests.post(self.nvai_url, headers=headers, json=payload)
            
            return ProcessorOutput(
                vector=response.json()["metadata"][0]["embedding"] if response.json()["metadata"] else None,
                metadata=response.json()["metadata"] if response.json()["metadata"] else None,
                error=response.reason
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
        
def pprint(result: ProcessorOutput):
    if result.vector:
        print(result.vector)
        print(f"向量生成成功，维度: {len(result.vector)}")
    else:
        print(result.error)
    return result
        

print_chain = RunnableLambda(pprint)


# 使用示例
def main():
    preprocessor = Preprocessor()

    convert_image_to_base64_chain = RunnableLambda(convert_image_to_base64)

    
    # 处理图片并直接获取向量
    (convert_image_to_base64_chain | preprocessor.preprocess_chain | print_chain).invoke("nv_preprocess/images/tiananmen.png")

    

if __name__ == "__main__":
    main()