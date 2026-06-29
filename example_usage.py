"""
Example usage of the LLM query client for students
"""
import os
import time
from dotenv import load_dotenv

from client.models import Query
from client.query import query_model

load_dotenv()

def query_model_with_retry(model_id: str, query: Query, max_retries: int = 3, delay: int = 4):
    for attempt in range(max_retries):
        try:
            response = query_model(model_id=model_id, query=query)
            return response
        except Exception as e:
            print(f"[Attempt {attempt + 1}/{max_retries}] Connection template hit an issue for Model {model_id}: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {delay} seconds before retrying")
                time.sleep(delay)
            else:
                print("Max retries reached. Raising the final exception")
                raise e

# Example usage for students
def main():
    # project_name = os.getenv("GCP_PROJECT_NAME")
    api_key = "YOUR-KEY-HERE"
    student_email = "YOUR-EMAIL-HERE"

    if not api_key or not student_email:
        print("Error: missing GEMINI_API_KEY or STUDENT_EMAIL in .env!")
        return

    # if not project_name or not student_email:
    #     print("Error: required environment variables not set!")
    #     print("Please:")
    #     print("1. Copy .env.template to .env")
    #     print("2. Edit .env to include GCP_PROJECT_NAME and STUDENT_EMAIL")
    #     print("3. Re-run this script")
    #     return
    
    # Create a query with conversation turns
    query = Query(turns=[
        {"user": "Hello! Can you help me understand transformers?"},
    ])
    
    # Query model A
    # response = query_model(
    #     model_id="A",
    #     query=query
    # )
    
    # print("=" * 100)
    # print(f"Model A Response")
    # print("=" * 100)
    # print(f"\n{response.text}\n")
    # print("=" * 100)
    # print(f"Cost: ${response.cost:.8f}")
    # print(f"Tokens used: {response.input_tokens} input, {response.output_tokens} output")
    # print("=" * 100)
    print("\nSending request to Model A")
    try:
        response = query_model_with_retry(model_id="A", query=query)
        
        print("=" * 100)
        print("Model A Response")
        print("=" * 100)
        print(f"\n{response.text}\n")
        print("=" * 100)
        print(f"Cost: ${response.cost:.8f}")
        print(f"Tokens used: {response.input_tokens} input, {response.output_tokens} output")
        print("=" * 100)
    except Exception as e:
        print(f"\nTest failed for Model A: {e}")
    
    # Multi-turn conversation example with model B
    conversation = Query(turns=[
        {"user": "What is attention in transformers?"},
        {"assistant": "Attention is a mechanism that allows the model to focus on relevant parts of the input when processing each token."},
        {"user": "Can you give me a simple example?"}
    ])
    
    # response2 = query_model(
    #     model_id="B",
    #     query=conversation
    # )
    
    # print("\n" + "=" * 100)
    # print(f"Model B Response")
    # print("=" * 100)
    # print(f"\n{response2.text}\n")
    # print("=" * 100)
    # print(f"Cost: ${response2.cost:.8f}")
    # print(f"Tokens used: {response2.input_tokens} input, {response2.output_tokens} output")
    # print("=" * 100)
    print("\nSending request to Model B")
    try:
        response2 = query_model_with_retry(model_id="B", query=conversation)
        
        print("\n" + "=" * 100)
        print("Model B Response")
        print("=" * 100)
        print(f"\n{response2.text}\n")
        print("=" * 100)
        print(f"Cost: ${response2.cost:.8f}")
        print(f"Tokens used: {response2.input_tokens} input, {response2.output_tokens} output")
        print("=" * 100)
        print("\nAPI authentication and query engine are working fine!")
    except Exception as e:
        print(f"\nTest failed for Model B: {e}")

if __name__ == "__main__":
    main()