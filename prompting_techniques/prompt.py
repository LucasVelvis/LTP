from abc import abstractmethod
from data import Data
from models.model import Model


class Prompt:
    """
    This is the prompt superclass.
    It has a standard text format, which is a string. And had subclasses, that implement
    separate prompting techniques.

    Parameters:
    - name: str, the name of the prompting technique
    - text: str, the text of the prompt
    - data: Data, the data object
    - model: Model, the model object
    """
    def __init__(self, name, text: str, data: Data, model: Model = None):
        self.name = name
        self.text = text
        self.data = data
        self.model = model
        self.additional_info = ""

    def __repr__(self) -> str:
        return self.get_standard_format(self.text)
    
    def get_standard_format(self, text: str) -> str:
        """ Standard format retrieved from the MAFALDA paper."""

        standard_format = f"""
Definitions:
• An argument consists of an assertion called
the conclusion and one or more assertions
called premises, where the premises are intended to establish the truth of the conclusion.
Premises or conclusions can be implicit in an
argument.
• A fallacious argument is an argument where
the premises do not entail the conclusion.
Text: "{self.get_prompt_context()}"
Based on the above text, determine whether the
following sentence is part of a fallacious argument
or not. If it is, indicate the type(s) of fallacy without providing explanations. The potential types of
fallacy include:
• non-fallacious
• hasty generalization
• causal oversimplification 
• Appeal to Ridicule 
• false dilemma 
• ad hominem 
• nothing
• ad populum 
• straw man 
• false causality 
• false analogy
• slippery slope
• appeal to fear 
• appeal to nature
• circular reasoning 
• appeal to (false) authority 
• appeal to worse problems 
• guilt by association
• equivocation 
• appeal to tradition 
• appeal to anger 
• appeal to positive emotion 
• tu quoque
• fallacy of division
• appeal to pity
• fallacy of relevance 
• intentional
• appeal to emotion
Sentence: "{text}"
{self.additional_info}
Output:
""" 
        return standard_format
        
    @abstractmethod
    def get_prompt_context(self) -> str:
        """ To be implemented by subclasses. Returns the context of the prompt specific to type of prompting technique. """
        pass