import json
PRICING={
    "gpt-40-mini":{"input":0.15,"output":0.6},
    "gpt-40":{"input":2.5,"output":.10},
    "text-embedding-3-small":{"input":0.02,"output":0.0},
    "openai/gpt-oss-120b":{"input":0.15,"output":0.6},
}

def estimate_cost(model,prompt_tokens,completion_tokens):
    pricing = PRICING.get(model)
    input_cost= prompt_tokens/1000000*pricing["input"]
    output_cost = completion_tokens/1000000*pricing["output"]
    return input_cost + output_cost

def log_api_call(messages,response,fn="data/api_log.jsonl"):
    log_entry={
        "timestamp" : response.model,
        "prompt": messages,
        "response": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "cost": estimate_cost(
            response.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        ),
        "finish_reason": response.choices[0].finish_reason 
    }
    with open(fn,"w") as f:
        f.write(json.dumps(log_entry)+"\n")
        return log_entry