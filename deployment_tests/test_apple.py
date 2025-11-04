from retrieve_results import get_response_from_agent
from llm_as_a_judge import judge
import pandas as pd
from tqdm import tqdm

deployment_url = "https://rag-agent-atri-53463e0a7b9652ac891bd5b274df0c13.us.langgraph.app"# Replace with your deployment URL



if __name__ == "__main__":
    
    data = pd.read_csv("/Users/atriviveksharma/Desktop/SafeIntelligence/LLM_experiments/datasets/FinDER/data/train.csv")
    
    responses = []
    
    
    apple_data = data[data.text.str.contains("AAPL", case=False)]
    
    correct = 0
    total = len(apple_data)

    for index, row in tqdm(apple_data.iterrows(), total=total):
        response = get_response_from_agent(deployment_url, row.text)
        verdict = judge(row.answer, response)
        print(f"Row {index}: {verdict}")
        
        if verdict.strip().upper() == "ACCEPT":
            correct += 1
            
        responses.append({
            "question": row.text,
            "expected_answer": row.answer,
            "model_response": response,
            "verdict": verdict
        })

    print(f"Accuracy: {correct / total * 100:.2f}%")
    
    results_df = pd.DataFrame(responses)
    results_df.to_csv("apple_deployment_results.csv", index=False)