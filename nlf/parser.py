import re

OPERATORS = {
    "is": "__iexact",
    "equals": "__iexact",
    "contains": "__icontains",
    "starts with": "__istartswith",
    "ends with": "__iendswith",
    "greater than": "__gt",
    "less than": "__lt",
}

def parse_query(query: str):
    """
    Convert NL query into structured components.
    Example: "get all users whose first name is aravind"
    Returns: {'model': 'user', 'filters': {'first_name__iexact': 'aravind'}}
    """
    query = query.lower()

    # Find model name
    model_match = re.search(r"(users|user|candidates|candidate)", query)
    model = model_match.group(0)[:-1] if model_match else None

    # Extract conditions
    filters = {}
    for op in OPERATORS:
        if op in query:
            parts = query.split(op)
            field_part, value_part = parts[0], parts[1]

            field_name = field_part.strip().split()[-1]  # crude extraction
            value = value_part.strip().split()[0]

            filters[f"{field_name}{OPERATORS[op]}"] = value
            break

    return {"model": model, "filters": filters}
