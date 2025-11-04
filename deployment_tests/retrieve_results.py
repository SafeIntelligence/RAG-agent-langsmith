
import requests
import json
import re


with open("/Users/atriviveksharma/Desktop/SafeIntelligence/LLM_experiments/langsmith_key.txt","r") as f:
    langsmith_api_key = f.read().strip()
    
def get_response_from_deployment(deployment_url: str, query: str):
    
    response = requests.post(
    f"{deployment_url}/runs/stream",
    headers={
        "Content-Type": "application/json",
        "X-Api-Key": langsmith_api_key
    },
    json={
        "assistant_id": "55c618b6-1afc-4901-8450-b40a771f0713",  # Name of agent. Defined in langgraph.json.
        "input": {
            "messages": [
                {
                    "role": "human",
                    "content": query
                }
            ]
        },
        # "stream_mode": "updates"
    })

    return response

def decode_response(response):

    # use the Response already in the notebook
    s = response.content.decode(errors="replace")

    # extract all lines that start with "data: "
    data_lines = re.findall(r"^data: (.*)", s, flags=re.MULTILINE)

    parsed = []
    for d in data_lines:
        d = d.strip()
        if not d:
            continue
        try:
            parsed.append(json.loads(d))
        except json.JSONDecodeError:
            # skip non-JSON data (e.g., heartbeat or plain text)
            continue

    # print(parsed)

    return parsed[-1]['messages'][-1]['content']

def get_response_from_agent(deployment_url: str, query: str):

    response = get_response_from_deployment(deployment_url, query)
    final_response = decode_response(response)
    return final_response