from src.models.ridge import Ridge
from src.models.mlp import MLP
from src.models.base_model import BaseModel


def create_model(name: str, **kwargs) -> BaseModel:
    """Simple factory method for models available"""

    models = {"ridge": Ridge, "mlp": MLP}
    try:
        return models[name.lower()](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(models.keys())}")
