import os
from langchain_together import Together
import settings


os.environ['TOGETHER_API_KEY'] = settings.TOGETHER_API_KEY

def create_llm():
    """
    Creata an instance of Nous-Hermes-2-Mistral-7B-DPO LLM using Together

    returns:
    - llm: An instance of Mistral 7B LLM
    """
    # Create llm
    llm = Together(
                    model="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                    temperature=1,
                    max_tokens=256,
                    top_k=1,
                    # together_api_key="..."
                )
    return llm


    
