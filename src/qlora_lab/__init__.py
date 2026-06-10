"""qlora-lab: a productionized QLoRA fine-tuning lab for structured extraction.

Decide -> data -> train -> serve -> evaluate -> integrate, each step a module you
can read in one sitting. See the README for the reading path.
"""
from . import agent, dataset, dpo, evaluate, predict, schema, serve, synth, train

__all__ = ["schema", "synth", "dataset", "predict", "evaluate", "train", "serve", "agent", "dpo"]
__version__ = "0.1.0"
