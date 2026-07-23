exact_cache = {}

def _normalize_key(q):
    if q is None:
        return q
    return q.strip().lower()

def get_exact_match(query):
    return exact_cache.get(_normalize_key(query))

def store_exact_match(query, answer):
    exact_cache[_normalize_key(query)] = answer

def reset():
    exact_cache.clear()