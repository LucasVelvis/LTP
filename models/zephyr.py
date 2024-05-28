from models.model import Model
from transformers import AutoTokenizer, AutoModelForCausalLM


class Zephyr(Model):
    """
    This class is used to handle the Falcon model.

    No parameters are needed.
    """
    def __init__(self):
        super().__init__(name="Falcon", model="HuggingFaceH4/zephyr-7b-beta")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(self.model)

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
