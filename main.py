from data import Data
from prompting_techniques.one_shot import OneShot

if __name__ == "__main__":
    # Load the data
    data = Data()
    
    # Get a prompt
    prompt, label = data[0]
    one_shot_prompt = OneShot(prompt, data)
    print(prompt)
    print(one_shot_prompt)