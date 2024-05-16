from transformers import AutoTokenizer, AutoModelForCausalLM

class FalconModel:
    def __init__(self):
        """
        Initializes both the tokenizer and model.
        """
        model_name = "tiiuae/falcon-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    def generate_response(self, prompt):
        """
        Generates a response to some given prompt
        """
        # Encode it
        input = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate the output
        output = self.model.generate(input, max_length=50)

        # Decode and return the output
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# Just a quick test (takes too long to run)
if __name__ == "__main__":
    handler = FalconModel()
    prompt = "What's up?"
    response = handler.generate_response(prompt)
    print("Response:", response)
