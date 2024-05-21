"""
Quick test model to not have to deploy the whole thing.
"""

import random
from models.model import Model


class TestModel(Model):
    """
    This class is used to handle the Test model.

    No parameters are needed.
    """
    def __init__(self):
        super().__init__("Test")
        self.latest_response = "This is a test response."

    def generate_response(self, prompt):
        """
        Generates a response to some given prompt
        """
        options = [
            "This is and ad hominem fallacy (23, 12).",
            "This is a strawman fallacy (12, 12).",
            "This is a false dichotomy fallacy.",
            "This is not a fallacy.",
            "Hi, how are you?"
        ]
        self.latest_response = random.choice(options)
        return self.latest_response