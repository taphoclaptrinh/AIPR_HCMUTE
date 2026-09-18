from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

@dataclass
class EmbeddingResult:
    vector: list[float]
    model: str = "text-embedding-3-small"
    token_count: int = 0                    
    created_at: datetime = field(default_factory=datetime.now)
    text: str = ""

result1 = EmbeddingResult(
    vector=[0.023, -0.045, 0.112, 0.067],
    model="text-embedding-3-small",
    token_count=10,
    text="What is RAG?"
)

result2 = EmbeddingResult(
    vector=[0.089, 0.001, -0.032, 0.055],
    model="text-embedding-3-small",
    token_count=8,
    text="Python programming"
)

results_dict = [asdict(result1), asdict(result2)]

for r in results:
    #convert datetime to string for JSON serialization
    r['created_at'] = r['created_at'].isoformat()

json_str= json.dumps(results, indent = 2)
print(json_str)

json_output = json.dumps(results_dict, indent=2, default=str)
print(json_output)