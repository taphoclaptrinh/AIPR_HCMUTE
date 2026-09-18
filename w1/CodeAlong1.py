import json
from datetime import datetime
with open(r"D:\Thanh\vscode\AIPR_HCMUTE-main\AIPR_HCMUTE-main\w1\data\chat_messages.json", "r") as f:
    data = json.load(f)

print(f"Loaded: {len(data)} messages")
assistant_msgs = [msg for msg in data if msg.get("role") == "assistant"]
print(f"Assistant messages: {len(assistant_msgs)}")
l = len(assistant_msgs)
print(f"Assistant messages: {l}")
for msg in assistant_msgs[:l]:
    print(f"{msg["role"]}: {msg["content"][:50]}...")

with open(r"D:\Thanh\vscode\AIPR_HCMUTE-main\AIPR_HCMUTE-main\w1\data\chat_messages_output.json", "w") as fo:
    json.dump(assistant_msgs, fo, indent=2)