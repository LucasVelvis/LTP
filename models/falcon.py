import torch
from models.model import Model
from transformers import AutoTokenizer, pipeline


class Falcon(Model):
    """
    This class is used to handle the Falcon model.

    No parameters are needed.
    """
    def __init__(self):
        super().__init__(name="Falcon", model="tiiuae/falcon-7b-instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate_response(self, prompt):
        """
        Generates a response to some given prompt
        """
        prompt = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
        sequences = self.pipeline(
            prompt,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return sequences[0]['generated_text']
