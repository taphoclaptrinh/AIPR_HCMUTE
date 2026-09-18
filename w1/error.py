from dotenv import load_dotenv
load_dotenv()
import os
import random
import time

from groq import (
    APIConnectionError,
    Groq,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    APIStatusError,
)
import json
from datetime import datetime, timezone

client = Groq(api_key="12345")

def call_with_retry_single(messages, model="openai/gpt-oss-20b", max_retries=3):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            log_entry = {
                "timestamp":response.model,
                "prompt":messages,
                "response":response.choices[0].message.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": estimate_cost(
                    response.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                    ),
                "finish_reason":response.choices[0].finish_reason
            }
            os.makedirs("data", exist_ok=True)
            with open("data/api_log.jsonl", "a") as f:
                f.write(json.dump(log_entry) + "\n")

            return response
        except RateLimitError as e:
            last_error = e
            if attempt < max_retries:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print("Max retries exceeded.")
                raise
        except APIConnectionError as e:
            last_error = e
            if attempt < max_retries:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"Connection error. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print("Max retries exceeded.")
                raise
        except (AuthenticationError, BadRequestError) as e:
            print(f"Fatal error: {e}. Not retrying.")
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "error": str(e),
                "success": False,
                "attempt": attempt + 1
            }
            os.makedirs("data", exist_ok=True)
            with open("data/api_log.jsonl", "a") as f:
                f.write(json.dumps(log_entry))
            raise
        except APIStatusError as e:
            last_error = e
            if 500 <= e.status_code < 600 and attempt < max_retries:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"Server Error {e.status_code}. Retrying...")
                time.sleep(delay)
            else:
                raise
    raise last_error or Exception("Unknown error")


messages = [{"role": "user", "content": "What is RAG"}]
try:
    response = call_with_retry_single(messages)
    print(response.choices[0].message.content)
except (AuthenticationError, BadRequestError):
    print("Program stopped because error cannot retry")