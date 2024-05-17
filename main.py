from data import Data
from prompting_techniques.one_shot import OneShot
from prompting_techniques.few_shot import FewShot
import sys

if __name__ == "__main__":
    # Load the data
    data = Data()

    prompt, label = data[0]
    
    if sys.argv[1] == "few_shot":
        # Get a few-shot prompt
        few_shot_prompt = FewShot(prompt, data)
        print(prompt)
        print(few_shot_prompt)
    elif sys.argv[1] == "one_shot":
        # Get a one-shot prompt
        one_shot_prompt = OneShot(prompt, data)
        print(prompt)
        print(one_shot_prompt)
