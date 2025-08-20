from django.apps import apps

MODEL_REGISTRY = {}

def register_model(model_name, model_class, field_map=None):
    """
    Registers a model for natural language filtering.
    
    :param model_name: str - alias for the model (e.g. 'user')
    :param model_class: Django model class
    :param field_map: dict - maps natural language to field names
    """
    MODEL_REGISTRY[model_name.lower()] = {
        "class": model_class,
        "fields": field_map or {}
    }

def get_model(model_name):
    return MODEL_REGISTRY.get(model_name.lower())
