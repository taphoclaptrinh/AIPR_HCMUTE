import json
#1
with open(r"D:\Thanh\vscode\AIPR_HCMUTE-main\AIPR_HCMUTE-main\w1\reviews.json", "r") as f:
    reviews = json.load(f)
#2
unique_tuples = set(tuple(r.items()) for i in reviews)
unique_reviews = [dict(t) for t in]
#3
reviews_with_empty_text = [r for r in reviews if r["text"] != ""]
#4
