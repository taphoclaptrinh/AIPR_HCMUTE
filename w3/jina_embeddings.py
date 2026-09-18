import requests

class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

def _wrap(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj

def jina_embeddings(
    input,
    api_key,
    model: str = "jina-embeddings-v3",
    task: str = "retrieval.passage",
    normalized: bool = True,
    dimensions: int | None = None,
    timeout: int = 30,
):
    if not api_key:
        raise ValueError("api_key is required (e.g. os.getenv('JINA_API_KEY'))")

    if isinstance(input, str):
        texts = [input]
    else:
        texts = list(input)

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "task": task,
        "normalized": normalized,
        "input": [{"text": t} for t in texts],
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()

    data = result.get("data", [])
    for i, item in enumerate(data):
        item.setdefault("index", i)

    return _wrap({
        "data": data,
        "model": result.get("model", model),
        "usage": result.get("usage", {}),
    })