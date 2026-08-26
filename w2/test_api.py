from groq import Groq
import os
from rich import print
from dotenv import load_dotenv
from api_helpers import estimate_cost, log_api_call


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
      {"role":"system","content":"You are an experienced python developer. You can explain code algorithms in the most simple way but never providing code."},
      {"role":"user","content":"1+1="}
    ],
    temperature = 0.2,
    max_tokens = 1000

)

print(response.choices[0].message.content)
print("===Metadata==")
print(f"model = {response.model}")
print(f"prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
print(f"reasoning: {response.choices[0].message.reasoning}")

usage = response.usage
input = (0.15/1000000)*usage.prompt_tokens
output = (0.6/1000000)*usage.completion_tokens

print(f"inpust cost: {input}")
print(f"output cost: {output}")

model = response.model
prompt_tokens = usage.prompt_tokens
completion_tokens = usage.completion_tokens
total_cost = estimate_cost(model,prompt_tokens,completion_tokens)
print(f"total cost: {total_cost}")



print(response)