
from abc import abstractmethod
from typing import List
import json


class Model:
    """
    This is the model superclass, to handle all common functionality between models.

    Parameters:
    - name: str, the name of the model.
    """

    def __init__(self, name: str, model: str = ""):
        self.name = name
        self.model = model
        self.latest_response: str = ""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

    def write_response(self, prompt: str, labels: List[str], prompting_technique: str):
        data = {
            "text": prompt.text,
            "response": self.latest_response,
            "labels": labels
        }
        json_data = json.dumps(data)
        with open(f"data/responses/{self.name}_{prompting_technique}.jsonl", "a") as f:
            f.write(json_data + "\n")