from .parser import parse_query
from .registry import get_model

def filter_query(query: str):
    parsed = parse_query(query)
    model_info = get_model(parsed["model"])

    if not model_info:
        raise ValueError(f"Model '{parsed['model']}' not registered.")

    model_class = model_info["class"]
    return model_class.objects.filter(**parsed["filters"])
