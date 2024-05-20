"""
Data format (sample):
{"text": "TITLE: Endless Ledge Skip Campaign for Alts POST: Reading everyone's comments has made me change my opinion on having an adventure mode like thing in PoE. I keep seeing if an adventure mode needs to exist then why not just start a char at 68. then a couple months after that we'll see post about how people want to start at 90 just so they can kill bosses. Then a few months later we'll see people asking for a\"creative\" mode so they can have access to all items in the game so they don't have to farm, and i dont think that sets a good example as a community for a game that we all love.\n", "labels": [[155, 588, "slippery slope"]], "comments": ["Slippery slope: P1 = poster, A = why not just start a char at 68, B = then a couple months after that we'll see post about how people want to start at 90 just so they can kill bosses, C = Then a few months later we'll see people asking for a\"creative\" mode so they can have access to all items in the game, D= so they don't have to farm", "Usage of a slippery slope for removing adventure modes."], "sentences_with_labels": "{\"TITLE: Endless Ledge Skip Campaign for Alts POST: Reading everyone's comments has made me change my opinion on having an adventure mode like thing in PoE.\": [[\"nothing\"]], \"I keep seeing if an adventure mode needs to exist then why not just start a char at 68.\": [[\"slippery slope\"]], \"then a couple months after that we'll see post about how people want to start at 90 just so they can kill bosses.\": [[\"slippery slope\"]], \"Then a few months later we'll see people asking for a\\\"creative\\\" mode so they can have access to all items in the game so they don't have to farm, and i dont think that sets a good example as a community for a game that we all love.\": [[\"slippery slope\"]]}"}
{"text": "Two of my best friends are really introverted, shy people, and they both have cats. That leads to me believe that most cat lovers are really shy.\n", "labels": [[84, 145, "hasty generalization"]], "comments": ["Based on two people only, you can't draw general conclusions.", "Hasty generalization: S=2 introverted friends, P=introverted people, C=are cat lovers"], "sentences_with_labels": "{\"Two of my best friends are really introverted, shy people, and they both have cats.\": [[\"nothing\"]], \"That leads to me believe that most cat lovers are really shy.\": [[\"hasty generalization\"]]}"}
{"text": "TITLE: There is a difference between a'smurf' and an'alt'. Please learn it and stop using them interchangeably. POST: Someone once told me they have an\"alt\" cause their main account was too high of rank to play with their friends. It's exactly the same as smurfing.\n", "labels": [[118, 265, "false analogy"]], "comments": ["False Analogy: X: Having an alt , Y: smurfing, P: Both involve having a secondary account.", "We removed the hasty gen", "the text may involve a \"False Equivalence\" fallacy. This is when someone incorrectly asserts that two or more things are equivalent, simply because they share some characteristics, despite the fact that there are also notable differences between them. In your example, the person is equating having an 'alt' account to play with friends of a lower rank with 'smurfing'. While both involve using a secondary account, the motivations and consequences may be different, so it's not necessarily accurate or fair to say they are \"exactly the same\".\n\nThis could be seen as a folse analogy too."], "sentences_with_labels": "{\"TITLE: There is a difference between a'smurf' and an'alt'.\": [[\"nothing\"]], \"Please learn it and stop using them interchangeably.\": [[\"nothing\"]], \"POST:\": [[\"nothing\"]], \"Someone once told me they have an\\\"alt\\\" cause their main account was too high of rank to play with their friends.\": [[\"false analogy\"]], \"It's exactly the same as smurfing.\": [[\"false analogy\"]]}"}
"""

import json
from torch.utils.data import Dataset
from random import shuffle
import re

"""
This class should preprocess the data by splitting it into:
- prompt
- labels, which is split into:
    - start index
    - end index
    - label name
"""
class Label:
    """
    Simple class to represent a label in the dataset.
    """
    def __init__(self, start: int, end: int, name: str):
        self.start = start
        self.end = end
        self.name = name

    def __repr__(self) -> str:
        return self.name + " (" + str(self.start) + ", " + str(self.end) + ")"

class Data(Dataset):
    """
    Standard dataset class for the data from the golden standard dataset.

    parameters:
    - (Optional) data_path: str, the path to the data file
    - (Optional) sample_size: int, the size of the sample to take from the data
    """
    def __init__(self, data_path="data/gold_standard_dataset.jsonl", sample_size=None):
        self.data_path = data_path
        self.data = self.load_data(data_path, sample_size=sample_size)

    def change_sample_size(self, sample_size: int):
        self.data = self.load_data(self.data_path, sample_size=sample_size)
    
    def load_data(self, data_path: str, sample_size: int = None):
        """ Load the data from the given path. (Possibly adjust for sample size, by randomly shuffling and taking the first n elements.) """
        with open(data_path, "r") as f:
                data = [json.loads(line) for line in f]
        if sample_size is not None:
            shuffle(data)
            data = data[:sample_size]
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        prompt = item["text"]
        # process the prompt (adjust for title, post, etc.)
        # TODO: decide whether this is necessary
        # prompt = re.sub(r"TITLE:|POST:|COMMENT:|SENTENCE:|\"|\\", "", prompt)
 
        labels = item["labels"]
        formatted_labels = []
        for label in labels:
            formatted_labels.append(Label(label[0], label[1], label[2]))
        
        return prompt, formatted_labels
    

if __name__ == "__main__":
    # Load the data
    data_path = "data/gold_standard_dataset.jsonl"
    data = Data(data_path)
    prompt, label = data[0]
    print(prompt)
    print(label)