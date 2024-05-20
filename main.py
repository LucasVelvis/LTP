from data import Data
from prompting_techniques.zero_shot import ZeroShot
from prompting_techniques.few_shot import FewShot
from models.test import TestModel
from experiment import Experiment
import sys

if __name__ == "__main__":
    # Default to one-shot if no argument is given
    if len(sys.argv) == 1:
        sys.argv.append("zero_shot")

    # Load the data
    data = Data()

    prompt, label = data[0]
    
    if sys.argv[1] == "few_shot":
        # Get a few-shot prompt
        few_shot_prompt = FewShot(prompt, data)
        print(prompt)
        print(few_shot_prompt)
    elif sys.argv[1] == "zero_shot":
        # Get a zero-shot prompt
        zero_shot_prompt = ZeroShot(prompt, data)
        print(prompt)
        print(zero_shot_prompt)
    elif sys.argv[1] == "experiment":
        model = TestModel
        experiment = Experiment(data, models=[model])
        experiment.run()

