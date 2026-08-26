from groq import Groq
import sys
import os
from dotenv import load_dotenv
from api_helpers import estimate_cost, log_api_call

# SYSTEM_INDICATORS = [
#     "bạn là", "dịch sang", "hãy đóng vai", "hãy trở thành", 
#     "you are", "act as", "role:", "ngôn ngữ:"
# ]

# def format_prompt(prompt:str):
#     prompt_lower = prompt.strip().lower()
#     is_system = any(p in prompt_lower for p in SYSTEM_INDICATORS)
#     if is_system:
#         return "system"

def ai_response(response, prompt, role):
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

def prompt_process():
    if len(sys.argv) < 3:
        print("Loi vui long truyen du tham so!")
        print("Cu pha dung: python independent_lab3.py <role> <prompt>")
        sys.exit(1)

    role = sys.argv[1]
    prompt = str(" ".join(sys.argv[2:]))
    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages = [{"role": role, "content": prompt}],
        temperature = 0.2,
        max_tokens = 1000
    )

    return ai_response(response=response, prompt=prompt, role=role)

#4. Save conversation to data/exploration_log.jsonl#
def save_conversation_to_jsonl(prompt, response, role, temperature):
    import json
    ai_content = response.choices[0].message.content
    reasoning = getattr(response.choices[0].message, "reasoning", None)
    usage = response.usage
    log_entry = {
        "model": response.model,
        "temperature": temperature,
        "role": role,
        "prompt": prompt,
        "response": ai_content,
        "reasoning": reasoning,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    }
    with open("data/exploration_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

#2. Send it twice (temperature = 0.0 , 0.8)#
def send_twice():
    temperatures = [0.0, 0.8]
    if len(sys.argv) < 3:
        print("Loi vui long truyen du tham so!")
        print("Cu pha dung: python independent_lab3.py <role> <prompt>")
        sys.exit(1)

    role = sys.argv[1]
    prompt = str(" ".join(sys.argv[2:]))
    load_dotenv()
    client = Groq()

    for temp in temperatures:
        print(f"\nPrompt voi temprerature = {temp}:")
        response = client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages = [{"role" : role, "content" : prompt}],
        temperature = temp,
        max_tokens = 1000
        )
        save_conversation_to_jsonl(prompt=prompt, response=response, role=role, temperature=temp)
        
        ai_response(response=response, prompt=prompt, role=role)
        print("\n")

if __name__ == "__main__":
    #prompt_process()
    send_twice()