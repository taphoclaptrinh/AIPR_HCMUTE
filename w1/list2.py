import json
with open("reviews.json", "r") as f:
    reviews = json.load(f)
print(f"Number of reviews: {len(reviews)}")
reviews_with_text = [r for r in reviews if r["text"] != ""]
top_rated = sorted(reviews_with_text, key=lambda r:r['rating'], reverse=True)[:3]
print(json.dumps(top_rated, indent=2))