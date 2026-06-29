import re
import json
import os
from typing import List, Dict

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model

import time

# You may find these constants useful for structuring the judge's output.
MODEL_E_PREFERED_TAG = "<MODEL_E_BETTER>"
MODEL_F_PREFERED_TAG = "<MODEL_F_BETTER>"
NO_PREFERENCE_FOUND_TAG = "<NO_PREFERENCE_FOUND>"


def load_alpaca_data() -> List[Dict[str, str]]:

    dataset = []
    with open("./data/alpaca_eval_first_30.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            dataset.append(example)

    return dataset

# def llm_judge_template(query: str, response_E: str, response_F: str) -> str:
#     """
#     Construct a prompt for an LLM judge to evaluate two model responses.

#     Args:
#         query: the question given to the two models (from AlpacaEval)
#         response_E: output from model E on query
#         response_F: output from model F on query
#     Returns:
#         Prompt for the LLM judge.
    
#     Consider: The judge is an LLM that will output free-form text. How will you 
#     design the prompt so that you can reliably determine which response it preferred?
#     Your llm_judge_template and extract_llm_judge_preference should work together.
#     """
#     # TODO complete for question 3b

#     pass

# def extract_llm_judge_preference(judge_output: str) -> str:
#     """
#     Extract the judge's preference from its output.

#     Args:
#         judge_output: the string sampled from the LLM judge.
#     Returns:
#         A string representing which response the judge preferred.
    
#     This function should work in tandem with your llm_judge_template design.
#     What if the judge's output is malformed or ambiguous?
#     """
#     # TODO complete for question 3b

#     pass

# def run_llm_judge_eval():
#     """
#     Run the LLM-as-a-judge evaluation comparing models E and F on AlpacaEval data.
#     Use model Z as the judge.
    
#     For each AlpacaEval instruction, you'll need responses from both models E and F,
#     then have the judge compare them.
    
#     Remember to save your results (model responses + judge outputs) - you will 
#     need them for Parts C and D.
#     """
#     # TODO complete for question 3b

#     pass
    
# def plot_model_output_lengths() -> None:
#     """
#     For Part D: Plot histograms of response lengths for preferred vs. not-preferred outputs.
#     """
#     # TODO complete for question 3d

#     pass

def llm_judge_template(query: str, response_E: str, response_F: str) -> str:
    return f"""You are an impartial judge evaluating two AI assistants.
User Prompt: {query}

[Model E Response]
{response_E}

[Model F Response]
{response_F}

Evaluate which response is better. If Model E is better, you MUST include the exact string "{MODEL_E_PREFERED_TAG}" at the end. If Model F is better, include "{MODEL_F_PREFERED_TAG}". If you cannot decide, include "{NO_PREFERENCE_FOUND_TAG}."
"""

def extract_llm_judge_preference(judge_output: str) -> str:
    if MODEL_E_PREFERED_TAG in judge_output: return "E"
    elif MODEL_F_PREFERED_TAG in judge_output: return "F"
    else: return "None"

def run_llm_judge_eval():
    dataset = load_alpaca_data()
    results = []
    win_E, win_F, ties = 0, 0, 0
    
    print("Running LLM Judge Evaluation...")
    # for item in tqdm(dataset):
    for item in tqdm(dataset[:3], desc="Judge Sandbox"):  # Use a smaller subset for sandbox testing
        instruction = item["instruction"]
        q = Query(turns=[{"user": instruction}])
        
        # try:
        #     res_E = query_model("E", q).text
        #     time.sleep(2)
        #     res_F = query_model("F", q).text
        #     time.sleep(2)
            
        #     judge_prompt = llm_judge_template(instruction, res_E, res_F)
        #     judge_q = Query(turns=[{"user": judge_prompt}])
        #     judge_res = query_model("Z", judge_q).text
        #     time.sleep(3)
            
        #     winner = extract_llm_judge_preference(judge_res)
        #     if winner == "E": win_E += 1
        #     elif winner == "F": win_F += 1
        #     else: ties += 1
                
        #     results.append({
        #         "instruction": instruction,
        #         "response_E": res_E,
        #         "response_F": res_F,
        #         "judge_rationale": judge_res,
        #         "winner": winner
        #     })
        # except Exception as e:
        #     print(f"Error skipping question: {e}")
        #     time.sleep(5)
        #     continue
        for attempt in range(10):
            try:
                res_E = query_model("E", q).text
                time.sleep(4) 
                
                res_F = query_model("F", q).text
                time.sleep(4) 
                
                judge_prompt = llm_judge_template(instruction, res_E, res_F)
                judge_q = Query(turns=[{"user": judge_prompt}])
                judge_res = query_model("Z", judge_q).text
                
                winner = extract_llm_judge_preference(judge_res)
                if winner == "E": win_E += 1
                elif winner == "F": win_F += 1
                else: ties += 1
                    
                results.append({
                    "instruction": instruction,
                    "response_E": res_E,
                    "response_F": res_F,
                    "judge_rationale": judge_res,
                    "winner": winner
                })
                break 
                
            except Exception as e:
                if "429" in str(e) or "503" in str(e):
                    print(f"\nSleeping for 30 seconds due to rate limit or service unavailable error (Attempt {attempt+1}/10)...")
                    time.sleep(30) 
                else:
                    raise e
        time.sleep(5)
        
    print(f"Wins: Model E={win_E}, Model F={win_F}, Ties/Unclear={ties}")
    with open("judge_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Results saved to judge_results.json")

def plot_model_output_lengths() -> None:
    try:
        with open("judge_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        return
        
    preferred_lengths, not_preferred_lengths = [], []
    for res in results:
        len_E, len_F = len(res["response_E"]), len(res["response_F"])
        if res["winner"] == "E":
            preferred_lengths.append(len_E)
            not_preferred_lengths.append(len_F)
        elif res["winner"] == "F":
            preferred_lengths.append(len_F)
            not_preferred_lengths.append(len_E)
            
    plt.figure(figsize=(10, 6))
    plt.hist(preferred_lengths, alpha=0.5, label='Preferred Output Lengths', color='blue', bins=10)
    plt.hist(not_preferred_lengths, alpha=0.5, label='Not Preferred Output Lengths', color='red', bins=10)
    plt.title("Output Lengths: Preferred vs Not Preferred")
    plt.legend()
    plt.savefig("output_lengths_histogram.png")
    plt.show()

if __name__=="__main__":

    # load_dotenv()

    ## Uncomment to run your code
    run_llm_judge_eval()
    plot_model_output_lengths()
