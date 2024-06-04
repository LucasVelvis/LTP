from data import Data
from models.model import Model
from prompting_techniques.prompt import Prompt
from random import shuffle
from fallacy_extraction import LEVEL_1_CLUSTERS, LEVEL_2_CLUSTERS, LEVEL_2_TO_LEVEL_1
from prompting_techniques.zero_shot import ZeroShot

class AutomaticCoT(Prompt):
    """
    The Automatic Chain of Thought (CoT) prompting technique.

    Parameters:
    - text: str, the text of the prompt
    - data: Data, the data object
    - model: Model, the model object
    - level: int, the level of fallacies to cluster the data into (1 or 2) (Optional, default is 2)
    """
    def __init__(self, text: str, data: Data, model: Model, level: int = 2):
        super().__init__("Automatic-CoT", text, data, model)
        self.level = level
        self.clusters = LEVEL_1_CLUSTERS if self.level == 1 else LEVEL_2_CLUSTERS
        self.clustered_data = {cluster: [] for cluster in self.clusters}

    def get_prompt_context(self) -> str:
        """ 
        Automatic CoT prompt context is some examples from the data with chain of thought reasoning.
        """
        self.cluster_data()
        self.demonstrations = self.generate_demonstrations()

        examples = []
        for cluster, demonstrations in self.demonstrations.items():
            if not demonstrations:
                continue

            example = demonstrations[0] + "\n"
            examples.append(example)
        return "\n".join(examples)
    
    def cluster_data(self) -> dict:
        """
        Cluster the questions into clusters (type of fallacies).
        There are two levels of clustering, corresponding to the level of fallacies in MAFALDA: level 1 and level 2.
        """
        for text, labels in self.data:
            for label in labels:
                cluster = label.name
                if self.level == 1:
                    try:
                        cluster = LEVEL_2_TO_LEVEL_1[cluster]
                    except KeyError:
                        print(f"Label {cluster} not found in the mapping.")
                        continue
                if cluster == "to clean":
                    continue
                self.clustered_data[cluster].append((text, label))

    def generate_demonstrations(self) -> str:
        """
        For all clusters choose a random representative question and generate a demonstration.
        This is done by Zero-Shot CoT; asking the model to think step by step.
        """
        demonstrations = {cluster: [] for cluster in self.clusters}
        for cluster, data in self.clustered_data.items():
            if not data:
                continue
            
            # Make sure the prompt does not occur in the demonstrations
            # And avoid infinite loop
            text = self.text
            iteration = 0
            while text == self.text and iteration < 10:
                shuffle(data)
                text, label = data[0]
                iteration += 1
            
            # If we did not find a suitable question, skip the cluster
            if iteration != 10:
                # We use a simple modification of Zero-Shot for Zero-Shot CoT
                zero_shot = ZeroShot(text, self.data, self.model)
                zero_shot.additional_info = "Think step by step."

                # Get the model response
                response = self.model.generate_response(str(zero_shot))
                
                # Add question + response to the demonstrations
                demonstrations[cluster].append(f"Question: {text}\nResponse: {response}")

        return demonstrations
