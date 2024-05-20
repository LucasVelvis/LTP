from data import Data
from typing import List
from util import get_all_models, get_all_prompting_techniques


class EvaluationFrameWork:
    """
    This class is used to evaluate the performance of the models.

    parameters:
    - models: List, the models to evaluate
    - prompting_techniques: List, the prompting techniques to evaluate
    """

    def __init__(self, models: List = None, prompting_techniques: List = None):
        # Default to all models and prompting techniques if none are given
        if models is None:
            models = get_all_models()
        if prompting_techniques is None:
            prompting_techniques = get_all_prompting_techniques()
        
        self.model = models
        self.prompting_techniques = prompting_techniques
        self.eval_data = Data()

    def evaluate(self):
        """
        Evaluates the performance of the models.
        """
        for model in self.models:
            for prompting_technique in self.prompting_techniques:
                path = f"data/{model.name}_{prompting_technique}.jsonl"
                try:
                    data = Data(data_path=path)
                except FileNotFoundError:
                    print(f"Data for model: {model}, prompting technique: {prompting_technique} not found, skipping.")
                    continue
                for prompt, labels in data:
                    self.score(self, model, prompting_technique, prompt, labels)


    def score(self, model, prompting_technique, prompt, labels):
        """
        Scores the model on the given prompt and labels.
        """
        pass
