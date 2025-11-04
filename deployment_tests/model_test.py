from pathlib import Path
from openai import OpenAI

API_KEY_PATH = Path("/Users/atriviveksharma/Desktop/SafeIntelligence/LLM_experiments/openrouter_key.txt")

SYSTEM_PROMPT = (
  "You are a medical student. answer the question as best as you can. "
)

with API_KEY_PATH.open("r", encoding="utf-8") as f:
  openrouter_api_key = f.read().strip()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=openrouter_api_key,
)

content = "what is aids"
model = "meta-llama/llama-3.2-1b-instruct"

completion = client.chat.completions.create(
    model=model,
    messages=[
        {
        "role": "system",
        "content": SYSTEM_PROMPT,
        },
        {
        "role": "user",
        "content": content
        },
    ],
    )

message = completion.choices[0].message

print(message.content)