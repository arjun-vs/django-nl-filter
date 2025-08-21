import ollama
import ast
from typing import List, Union, Type, Optional
from django.db import models
from django.contrib.contenttypes.fields import GenericForeignKey

def get_models_schema(models_list: List[Type[models.Model]]) -> str:
    """
    Generate a recursive schema for Django models, including fields and relations.
    Handles GenericForeignKey fields appropriately.
    """
    schema_parts = []
    visited_models = set()

    def add_model_schema(model):
        if model in visited_models:
            return
        visited_models.add(model)
        schema_parts.append(f"Model: {model.__name__}")
        for field in model._meta.get_fields(include_hidden=True):
            if isinstance(field, GenericForeignKey):
                schema_parts.append(f"  - {field.name}: GenericForeignKey (references any model via ContentType, use {field.name}__field_name for filtering)")
            elif hasattr(field, 'related_model') and field.related_model:
                schema_parts.append(f"  - {field.name}: ForeignKey to {field.related_model.__name__} (access via {field.name}__field_name)")
                add_model_schema(field.related_model)
            else:
                schema_parts.append(f"  - {field.name}: {field.get_internal_type()}")
        schema_parts.append("")

    for model in models_list:
        add_model_schema(model)
    return "\n".join(schema_parts)

def nl_to_orm(
    models: Union[Type[models.Model], List[Type[models.Model]]],
    nl_query: str,
    ollama_model: str = "mistral:instruct",
    custom_system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 500,
    validate_code: bool = True,
) -> str:
    """
    Convert natural language to advanced Django ORM query strings using Ollama.

    Supports all Django QuerySet techniques:
    - Filters: __gt, __lte, __in, __contains, etc.
    - Complex logic: Q objects (e.g., Q(field__gt=100) | Q(other=True))
    - Field ops: F expressions (e.g., F('score') + 1)
    - Subqueries: Subquery, Exists, OuterRef, When/Case
    - Aggregations: annotate/aggregate with Avg/Sum/Max/Min/Count
    - Optimizations: select_related, prefetch_related
    - Field selection: values, values_list, only, defer
    - Mutations: update, delete
    - Advanced: Func, date_trunc, raw SQL (sparingly)

    Args:
        models: Model(s) to query (includes related models).
        nl_query: Natural language input (e.g., "candidates with score > attempts or in active stages").
        ollama_model: Ollama model (e.g., "codellama:13b-instruct").
        custom_system_prompt: Custom prompt with {schema}, {nl_query}, {start_model}.
        temperature: For LLM determinism.
        max_tokens: For output length.
        validate_code: Check syntax of generated code.

    Returns:
        Django ORM query string, e.g., "from django.db.models import Q, F; Candidate.objects.filter(Q(score__gt=F('attempts')))"

    Raises:
        ValueError: If validate_code=True and syntax is invalid.

    Examples:
        query = nl_to_orm(Candidate, "candidates with score > 100 and name like 'John%'")
        query = nl_to_orm([Candidate, Stage], "annotate candidates with count of active stages")
        query = nl_to_orm(Candidate, "update scores to score + 5 where failed=True")
    """
    if not isinstance(models, list):
        models = [models]

    schema = get_models_schema(models)
    start_model = models[0].__name__

    if custom_system_prompt:
        system_prompt = custom_system_prompt.format(schema=schema, nl_query=nl_query, start_model=start_model)
    else:
        system_prompt = f"""You are an expert in Django ORM queries.
Given this schema (with relations):

{schema}

Convert the natural language to a valid Django ORM query string.
- Start with {start_model}.objects
- Use ALL QuerySet methods as needed: filter/exclude/get/all/order_by/distinct/values/values_list/only/defer/update/delete
- Advanced lookups: __gt/__lte/__in/__range/__contains/__startswith/__year
- Complex logic: Import Q (e.g., Q(field__gt=100) & ~Q(other=False))
- Field ops: Import F (e.g., filter(score__gt=F('attempts')))
- Subqueries: Import Subquery/Exists/OuterRef/When/Case
- Aggregates: Import Avg/Sum/Max/Min/Count; use annotate/aggregate
- Relations: Joins via __, select_related/prefetch_related
- Other: dates/date_trunc, Func, raw SQL (last resort)
- Mutations: Use .update/.delete if implied
- Imports: Prefix with 'from django.db.models import ...' for Q/F/Subquery/Avg/etc.
- For GenericForeignKey: Use content_type and object_id fields for filtering (e.g., filter(content_type__model='modelname', object_id__in=[...]))
- Output ONLY the Python code string, executable as-is."""

    response = ollama.chat(
        model=ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": nl_query},
        ],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    )

    generated_query = response['message']['content'].strip()
    if generated_query.startswith("```python"):
        generated_query = generated_query.split("```python")[1].split("```")[0].strip()
    elif generated_query.startswith("`"):
        generated_query = generated_query.strip("`").strip()

    if validate_code:
        try:
            ast.parse(generated_query)
        except SyntaxError as e:
            raise ValueError(f"Invalid query syntax: {e}\nQuery: {generated_query}")

    return generated_query