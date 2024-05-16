from prompting_techniques.prompt import Prompt

class OneShot(Prompt):
    """
    The one-shot prompt technique.

    parameters:
    - text: str, the text of the prompt
    - data: Data, the data object
    """

    def get_prompt_context(self) -> str:
        """ One-shot prompt context (is nothing)"""
        return self.text