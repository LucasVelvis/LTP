from prompting_techniques.prompt import Prompt
from models.model import Model
import json
from random import shuffle
import random
from typing import List

class GeneratedKnowledge(Prompt):
    """
    The generated knowledge prompt technique.

    parameters:
    - text: str, the text of the prompt
    - data: Data, the data object
    - model: Model, the model object
    """
    def __init__(self, text: str, data, model: Model):
        super().__init__("generated-knowledge", text, data, model)
        few_shot_examples = self.retrieve_few_shot_examples()
        self.generate_knowledge(few_shot_examples)

    def get_prompt_context(self) -> str:
        """ Generated knowledge prompt context"""
        context = f"input: {self.text}\n knowledge: {self.knowledge}"
        return context
    

    def retrieve_few_shot_examples(self):
        """Retrieve few-shot examples used as context for new knowledge generation."""
        # Get the data from the Generated Knowledge Prompting paper dataset
        path = "data/knowledge_gpt3.dev.csqa.json"
        with open(path, "r") as f:
            data = json.load(f)

        # Get the few-shot examples
        shuffle(data)
        examples = []
        for i in range(5):
            input = data[i]["query"]
            knowledge = random.choice(data[i]["knowledges"])
            examples.append(f"input: {input}\n knowledge: {knowledge}\n")
        return examples
    
    def generate_knowledge(self, few_shot_examples: List[str]):
        """
        Use the few-shot examples to generate new knowledge.
        """
        # Add the text to the few-shot examples
        formatted_text = f"input: {self.text}\n knowledge: \n"
        few_shot_examples.append(formatted_text)

        # Generate new knowledge
        prompt = "\n".join(few_shot_examples)
        response = self.model.generate_response(prompt)

        self.knowledge = response

        