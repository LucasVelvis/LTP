
from abc import abstractmethod
from typing import List


class Model:
    """
    This is the model superclass, to handle all common functionality between models.

    Parameters:
    - name: str, the name of the model.
    """

    def __init__(self, name: str):
        self.name = name
        self.latest_response: str = ""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

    def write_response(self, prompt: str, labels: List[str], prompting_technique: str):
        with open(f"data/responses/{self.name}_{prompting_technique}.jsonl", "a") as f:
            f.write(f'{{"text": "{prompt.text}", "response": "{self.latest_response}", "labels": {labels}}}\n')