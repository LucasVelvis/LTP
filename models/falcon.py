import torch
from models.model import Model
from transformers import AutoTokenizer, AutoModelForCausalLM


class Falcon(Model):
    """
    This class is used to handle the Falcon model.

    No parameters are needed.
    """
    def __init__(self):
        super().__init__(name="Falcon", model="HuggingFaceH4/zephyr-7b-beta")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def generate_response(self, prompt):
        """
        Generates a response to some given prompt
        """
        # Encode it
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to("cuda")

        # Generate the output
        outputs = self.model.generate(
            inputs,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and return the output
        self.latest_response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        return self.latest_response
