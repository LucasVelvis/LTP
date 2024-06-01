"""
Here the experiment is defined, which consists of running all prompting techniques on all models.
"""
from data import Data
from typing import List
from util import get_all_models, get_all_prompting_techniques
from fallacy_extraction import extract_fallacies
from models.model import Model
from prompting_techniques.prompt import Prompt


class Experiment:
    """
    This class is used to run the experiment.

    parameters:
    - models: List, the models to collect data for
    - prompting_techniques: List, the prompting techniques to collect data for
    """

    def __init__(self, data: Data = Data(), models: List[Model] = None, prompting_techniques: List[Prompt] = None):
        # Default to all models and prompting techniques if none are given
        if models is None:
            models = get_all_models()
        if prompting_techniques is None:
            prompting_techniques = get_all_prompting_techniques()

        self.data = data
        self.models = models
        self.prompting_techniques = prompting_techniques

    def run(self):
        """
        Runs the experiment for the chosen models and prompting techniques.
        """
        for model in self.models:
            print(f"Running experiment for model: {model.name}")
            for prompting_technique in self.prompting_techniques:
                # Filler prompt to log and clear existing data
                prompt = prompting_technique(text="Filler", data=self.data, model=model)
                print(f"Running experiment {model.name} for prompting technique: {prompt.name}")
                open(f"data/responses/{model.name}_{prompt.name}.jsonl", "w").close()

                # Loop over data
                for text, labels in self.data:
                    # Copy data to avoid modifying the original
                    # data = self.data.copy()

                    # Generate prompt
                    prompt = prompting_technique(text=text, data=self.data, model=model)

                    # Pass to model
                    response = model.generate_response(str(prompt))

                    # Extract fallacies from response
                    fallacies = extract_fallacies(response)

                    # Model.write_response
                    model.write_response(prompt=prompt, labels=fallacies, prompting_technique=prompt.name)