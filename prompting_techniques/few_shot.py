from data import Data
from prompting_techniques.prompt import Prompt
from random import shuffle
from models.model import Model


class FewShot(Prompt):
    """
    The few-shot prompting technique.

    parameters:
    - text: str, the text of the prompt
    - data: Data, the data object
    - model: Model, the model object
    - num_examples: int, the number of examples to show (Optional, default is 5)
    """
    def __init__(self, text: str, data: Data, model: Model, num_examples: int = 5):
        super().__init__("Few-Shot", text, data, model)
        self.num_examples = num_examples

    def get_prompt_context(self) -> str:
        """ Few-shot prompt context is some examples from the data."""
        examples = []
        shuffle(self.data.data)
        for i in range(self.num_examples):
            prompt, labels = self.data[i]
            if prompt != self.text: # Do not include the prompt itself if it is sampled
                output = [f"{labels.name} ({labels.start}, {labels.end})" for labels in labels]
                labeled_prompt = f"text: {prompt}\n output: {output}"
                examples.append(labeled_prompt)
        return "\n".join(examples)
