from openai import OpenAI
from pprint import pprint

# API_KEY = "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"
# BASE_URL = "https://openrouter.ai/api/v1"

API_KEY = "sk-rQjg5BwSXPNiltm4JaKKdaE3LYO6OUY3ce3FqymKpWEYXf0Z"
BASE_URL = "https://aifast.site/v1"

API_KEY = "sk-S9iP8pIdbK8XD4Hojm9g5rx1Rano3Muo8tsTFexIT6oJCuRV"
BASE_URL = "https://api.n1n.ai/v1"

MODEL = "gpt-5"
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

response = client.chat.completions.create(
    model=MODEL,
    messages=MESSAGES,
    response_format=None,
    temperature=0,
    max_tokens=2048,
    extra_body={
        'provider': {
            'order': ['openai', 'azure'],
        },
    }
)

pprint(response)