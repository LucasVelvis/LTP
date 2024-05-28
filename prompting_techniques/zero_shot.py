from prompting_techniques.prompt import Prompt
from models.model import Model


class ZeroShot(Prompt):
    """
    The Zero-shot prompt technique.

    parameters:
    - text: str, the text of the prompt
    - data: Data, the data object
    - model: Model, the model object
    """
    def __init__(self, text: str, data, model: Model):
        super().__init__("Zero-Shot", text, data, model)

    def get_prompt_context(self) -> str:
        """ Zero-shot prompt context (is nothing)"""
        return self.text
