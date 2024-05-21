import torch
from transformers import pipeline


class Zephyr(Model):
    """
    This class is used to handle the Falcon model.

    No parameters are needed.
    """
    def __init__(self):
        super().__init__("Zephyr")
        model_name = "HuggingFaceH4/zephyr-7b-beta"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, prompt):
        """
        Generates a response to some given prompt
        """
        # Encode it
        input = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate the output
        output = self.model.generate(input, max_length=50)

        # Decode and return the output
        self.latest_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self.latest_response
