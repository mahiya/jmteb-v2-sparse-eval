"""JMTEB v2 retrieval evaluation pipeline (reusable modules).

Public API:
    setup_environment(...)   -- prepare JMTEB checkout, packages, env vars, drive
    build_model(...)         -- load dense or sparse retrieval model
    get_target_tasks(...)    -- resolve task names to JMTEB task objects
    run_evaluation(...)      -- two-level try/except evaluation loop
"""

from .env_setup import setup_environment
from .model_loader import build_model
from .tasks import get_target_tasks
from .evaluator import run_evaluation

__all__ = [
    "setup_environment",
    "build_model",
    "get_target_tasks",
    "run_evaluation",
]
