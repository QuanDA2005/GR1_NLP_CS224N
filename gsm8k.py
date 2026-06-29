import re
import json
from typing import Dict
import os

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model

import time
from google.genai.errors import APIError

INVALID_ANS = "[invalid]"

def standard_prompt_template(question: str) -> str:
    """
    Converts a gsm8k question into a standard model input

    Args:
        question: gsm8k question.
    Returns:
        prompt for a model to answer input question.
    """

    prompt = f"""Output a numerical answer to the following problem with two or fewer steps of reasoning. Output your numerical
answer as the only line of your output in the format "#### <numerical_answer>."

Problem: {question}
""".strip()

    return prompt

def standard_output_extractor(model_generation: str) -> str:
    """
    Extracts the string answer from a model generation, assuming it was prompted 
    using a prompt from `standard_prompt_template`.

    Args:
        model_generation: the string generation from the model
    Returns:
        String representing the numerical output of the model for the question, or "[invalid]" if
            no output can be extracted.
    """

    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    match = ANS_RE.search(model_generation)

    if match:
        match_str: str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


# ------------------------------------------- #
# TODO For you to fill in 
# ------------------------------------------- #



# def eval_model_on_gsm8k() -> None:
#     """
#     Benchmark models A and B on the GSM8K dataset using the standard prompt template.
    
#     See example_usage.py for how to query models and handle responses.
#     The data file (gsm8k_first_100.jsonl) contains 'question' and 'numerical_answer' fields.
    
#     Think about: What metric will you use to evaluate performance? How will you 
#     handle cases where the model's output cannot be parsed?
#     """
#     # TODO complete for question 2bi

#     pass



# def superior_prompt_template(question: str) -> str:
#     """
#     Design your own prompt template that outperforms standard_prompt_template on model A.
    
#     Args:
#         question: gsm8k question.
#     Returns:
#         Your improved prompt for the model.
    
#     Look at standard_prompt_template() to understand the baseline approach. What 
#     aspects of how you prompt the model might affect its reasoning or accuracy?
    
#     NOTE: Your prompt must still produce output in the "#### <answer>" format
#     so that standard_output_extractor() can parse the response.
#     """
#     # TODO complete for question 2bii

#     pass

# def eval_model_on_gsm8k_with_improved_prompt() -> None:
#     """
#     Evaluate model A using your superior_prompt_template.
#     """
#     # TODO complete for question 2bii

#     pass

def eval_model_on_gsm8k() -> None:
    with open("data/gsm8k_first_100.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    correct_A, correct_B = 0, 0
    total = len(data)
    
    print("Evaluating Models A and B on GSM8K...")
    for item in tqdm(data, desc="Evaluating"):
        question = item["question"]
        true_ans = str(item["numerical_answer"])
        prompt = standard_prompt_template(question)
        query = Query(turns=[{"user": prompt}])
        
        for attempt in range(5):
            try:
                res_A = query_model(model_id="A", query=query)
                if standard_output_extractor(res_A.text) == true_ans:
                    correct_A += 1
                break
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    time.sleep(15)
                else:
                    raise e
        
        time.sleep(2) 

        for attempt in range(5):
            try:
                res_B = query_model(model_id="B", query=query)
                if standard_output_extractor(res_B.text) == true_ans:
                    correct_B += 1
                break
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    time.sleep(5)
                else:
                    raise e
                    
        time.sleep(2)
            
    print(f"Model A Accuracy: {correct_A}/{total} ({(correct_A/total)*100}%)")
    print(f"Model B Accuracy: {correct_B}/{total} ({(correct_B/total)*100}%)")

# def superior_prompt_template(question: str) -> str:
#     return f"""You are an expert mathematician. Solve the problem step by step to ensure accuracy.
# At the very end of your response, output a single line with just the numerical answer in the format "#### <numerical_answer>". Do not include commas.

# Problem: {question}
# """

def superior_prompt_template(question: str) -> str:
    return f"""You are an expert mathematician. Solve the problem step by step to ensure accuracy.
At the very end of your response, output your final numerical answer in the exact format "#### <numerical_answer>". 

CRITICAL RULES:
1. The line containing "#### <numerical_answer>" must be the absolute LAST line.
2. DO NOT put any period (.), spaces, or punctuation marks after the number. Example: "#### 72" is correct, "#### 72." is WRONG.

Problem: {question}
"""

def eval_model_on_gsm8k_with_improved_prompt() -> None:
    with open("data/gsm8k_first_100.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # correct_A = 0
    # total = len(data)
    sandbox_data = data[:2] 
    correct_A = 0
    total = len(sandbox_data)
    
    print("Evaluating Model A with Superior Prompt...")
    # for item in tqdm(data, desc="Evaluating Improved"):
    for item in tqdm(sandbox_data, desc="Evaluating Improved Sandbox"):
        question = item["question"]
        true_ans = str(item["numerical_answer"])
        prompt = superior_prompt_template(question)
        query = Query(turns=[{"user": prompt}])
        
        for attempt in range(10):
            try:
                res_A = query_model(model_id="A", query=query)
                print(f"\nDEBUG QUESTION")
                print(f"True answer in file: '{true_ans}'")
                print(f"Extracted string from Grader: '{standard_output_extractor(res_A.text)}'")
                print(f"Entire text generated by the model:\n{res_A.text}")
                if standard_output_extractor(res_A.text) == true_ans:
                    correct_A += 1
                break
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    print(f"\nError (Trying again, attempt {attempt+1}/10)...")
                    time.sleep(15)
                else:
                    raise e
        time.sleep(3)
            
    print(f"Model A (Improved) Accuracy: {correct_A}/{total} ({(correct_A/total)*100}%)")

if __name__=="__main__":

    # load_dotenv()

    ## Uncomment to run your code
    # eval_model_on_gsm8k()
    eval_model_on_gsm8k_with_improved_prompt()