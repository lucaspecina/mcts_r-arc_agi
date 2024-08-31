from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from langchain_ollama import ChatOllama

import os
import openai
from dotenv import load_dotenv

load_dotenv()

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
OPENAI_BASE_URL = None
OLLAMA_BASE_URL = "http://localhost:11434/v1"


def get_openai_api_key() -> str:
    # return os.environ["OPENAI_API_KEY"]
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key is not defined.")


def get_anthropic_api_key() -> str:
    return os.environ["ANTHROPIC_API_KEY"]

def get_fireworks_api_key() -> str:
    return os.environ["FIREWORKS_API_KEY"]

def get_ollama_api_key() -> str:
    return 'ollama'


def _get_openai_client(base_url: str | None = None) -> openai.Client:
    if base_url == FIREWORKS_BASE_URL:
        api_key = get_fireworks_api_key()
    elif base_url == OLLAMA_BASE_URL:
        api_key = get_ollama_api_key()
    else:
        api_key = get_openai_api_key()
    return openai.Client(api_key=api_key, base_url=base_url)


def openai_chat_completion(
    messages: list[ChatCompletionMessageParam],
    model: str,
    base_url: str | None = None,
    temperature: float = 0.8,
    **kwargs,
) -> ChatCompletion:
    client = _get_openai_client(base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response


# TODO: OLLAMA implementation (not tested)
def ollama_chat_completion(
    messages: list[ChatCompletionMessageParam],
    model: str,
    temperature: float = 0.8,
    **kwargs,
):
    llm = ChatOllama(model=model, temperature=temperature, max_tokens=4000)
    response = llm.invoke(
        {"llm_name": "llama3.1", "messages": messages},
    )
    return response