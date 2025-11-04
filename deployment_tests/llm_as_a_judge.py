from pathlib import Path

from openai import OpenAI

API_KEY_PATH = Path("/Users/atriviveksharma/Desktop/SafeIntelligence/LLM_experiments/openrouter_key.txt")
JUDGE_SYSTEM_PROMPT = (
  "You are an impartial judge scoring the alignment between a ground-truth answer "
  "and a candidate response. Consider factual accuracy and completeness. "
  "Only respond with a binary verdict: 'ACCEPT' if the candidate is accurate and complete, "
  "'REJECT' otherwise."
)

with API_KEY_PATH.open("r", encoding="utf-8") as f:
  openrouter_api_key = f.read().strip()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=openrouter_api_key,
)


def judge(ground_truth: str, generated_content: str, model: str = "openai/gpt-4o") -> str:
  """Call the judge model and return its textual verdict."""
  completion = client.chat.completions.create(
    model=model,
    messages=[
      {
        "role": "system",
        "content": JUDGE_SYSTEM_PROMPT,
      },
      {
        "role": "user",
        "content": (
          "Ground truth:\n"
          f"{ground_truth}\n\n"
          "Generated content:\n"
          f"{generated_content}"
        ),
      },
    ],
  )

  message = completion.choices[0].message
  if message.content is None:
    raise ValueError("Judge model returned no content.")

  return message.content


if __name__ == "__main__":
  example_ground_truth = "The capital of France is Paris."
  example_generated_content = "France's capital city is Paris."
  print(judge(example_ground_truth, example_generated_content))
