from data import Data
from prompting_techniques.zero_shot import ZeroShot
from prompting_techniques.few_shot import FewShot
from prompting_techniques.automatic_cot import AutomaticCoT
from prompting_techniques.generated_knowledge import GeneratedKnowledge
from models.test import RandomModel
from models.falcon import Falcon
from models.zephyr import Zephyr
from experiment import Experiment
from evaluation import EvaluationFrameWork
import sys


if __name__ == "__main__":
    # Default to one-shot if no argument is given
    if len(sys.argv) == 1:
        sys.argv.append("complete")

    # Load the data
    data = Data()
    model = RandomModel()

    prompt, label = data[0]
    
    if sys.argv[1] == "few_shot":
        # Get a few-shot prompt
        few_shot_prompt = FewShot(prompt, data, model)
        print(prompt)
        print(few_shot_prompt)
    elif sys.argv[1] == "zero_shot":
        # Get a zero-shot prompt
        zero_shot_prompt = ZeroShot(prompt, data, model)
        print(prompt)
        print(zero_shot_prompt)
    elif sys.argv[1] == "auto_cot":
        # Get an automatic CoT prompt
        auto_cot_prompt = AutomaticCoT(prompt, data, model)
        print(prompt)
        print(auto_cot_prompt)
    elif sys.argv[1] == "gen_knowledge":
        # Get a generated knowledge prompt
        gen_knowledge_prompt = GeneratedKnowledge(prompt, data, model)
        print(prompt)
        print(gen_knowledge_prompt)
    elif sys.argv[1] == "experiment":
        # Just run the experiment to get the data
        experiment = Experiment(data, models=[model])
        experiment.run()
    elif sys.argv[1] == "complete":
        # Run the experiment and evaluate the models
        model_falcon = Falcon()
        model_zephyr = Zephyr()
        experiment = Experiment(data, models=[model_falcon, model_zephyr])
        experiment.run()
        
        evaluation_framework = EvaluationFrameWork(models=[model_falcon, model_zephyr])
        evaluation_framework.evaluate()
        evaluation_framework.plot()

