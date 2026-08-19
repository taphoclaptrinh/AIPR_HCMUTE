import json
#1. Load reviews json
with open(r"E:\Thanh\w1\reviews.json","r") as f:
    reviews = json.load(f)
#2. Deduplicate by review_id
#------------------------cach 1-----------------------
unique_tuples = {tuple(d.items()) for d in reviews}
cleaned_reviews_tuple = list(dict(t) for t in unique_tuples)
#print(json.dumps(cleaned_reviews_tuple, indent=2))
#------------------------cach 2-----------------------

"""
print("""


""")
"""

unique_set = set()
cleaned_reviews_set = []
for t in reviews:
    if t["review_id"] not in unique_set:
        cleaned_reviews_set.append(t)
        unique_set.add(t["review_id"])

#print(json.dumps(cleaned_reviews_set,indent=2))

#3. Filter out reviews with empty text
cleaned_reviews = cleaned_reviews_set.copy()
cleaned_reviews = [t for t in cleaned_reviews if t["text"] != ""]
#print(json.dumps(cleaned_reviews,indent=2))

#4. Top 5 products by average rating
stats = {}
for p in reviews:
    if p["product"] not in stats:
        stats[p["product"]] = {"total_rating" : 0, "count" : 0}
    stats[p["product"]]["total_rating"] += p["rating"]
    stats[p["product"]]["count"] += 1
#stats = {"Laptop", "total_rating" : 0, "count" : 0}
avarage_rating = []
for prod, stat in stats.items():
    avarage_rating.append({"product" : prod, "rating" : stat["total_rating"] / stat["count"]})

avarage_rating.sort(key=lambda r: r["rating"],reverse=True)
top_5_products = avarage_rating[:5]
#5. Save (5) to top_5_products.json
with open(r"E:\Thanh\w1\top_5_products.json", "w") as fl:
    json.dump(top_5_products, fl, indent=2)