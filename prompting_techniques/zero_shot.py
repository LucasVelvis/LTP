from prompting_techniques.prompt import Prompt

class ZeroShot(Prompt):
    """
    The Zero-shot prompt technique.

    parameters:
    - text: str, the text of the prompt
    - data: Data, the data object
    """
    def __init__(self, text: str, data):
        super().__init__("zero-shot", text, data)

    def get_prompt_context(self) -> str:
        """ Zero-shot prompt context (is nothing)"""
        return self.text