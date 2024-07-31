import json
from asyncio import sleep

from openai import AsyncOpenAI
<<<<<<< HEAD
import re
=======

from ..logger import logger
from .client import Client


>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
class Local(Client):
    provider = "vllm"

    def __init__(self, model: str, base_url="http://localhost:8000/v1"):
        super().__init__(model)
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=None)
        self.model = model

    async def generate(
        self, prompt: str, raw: bool = False, max_retries: int = 2, **kwargs
    ) -> str:
        """
        Wrapper method for vLLM post requests.
        """

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model, messages=prompt, **kwargs
                )
<<<<<<< HEAD
                if raw:
                    return response
                
                processed_response = self.postprocess(response)
                
                if schema is not None:
                    match = re.search(
                        r'\{[^{}]*\}', 
                        processed_response, re.DOTALL
                    )
                    if match is not None:
                        processed_response = json.loads(match.group())
                    else:
                        raise RuntimeError("Failed to extract JSON response.")
                    
                return processed_response
            
=======

                return response if raw else self.postprocess(response)

>>>>>>> d3b81f50d676295bd88d0e499194b44c40c02b13
            except json.JSONDecodeError:
                logger.warning(
                    f"Attempt {attempt + 1}: Invalid JSON response, retrying..."
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")

            await sleep(1)

        logger.error("All retry attempts failed.")
        raise RuntimeError("Failed to generate text after multiple attempts.")

    def postprocess(self, response: dict) -> str:
        """
        Postprocess the response from the API.
        """
        return response.choices[0].message.content
