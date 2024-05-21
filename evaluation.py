from data import Data
from typing import List
from util import get_all_models, get_all_prompting_techniques
import matplotlib.pyplot as plt


class EvaluationFrameWork:
    """
    This class is used to evaluate the performance of the models.

    parameters:
    - models: List, the models to evaluate
    - prompting_techniques: List, the prompting techniques to evaluate
    """

    def __init__(self, models: List = None, prompting_techniques: List = None):
        # Default to all models and prompting techniques if none are given
        if models is None:
            models = get_all_models()
        if prompting_techniques is None:
            prompting_techniques = get_all_prompting_techniques()
        
        self.models = models
        self.prompting_techniques = prompting_techniques
        self.gold_standard = Data()

        # Metrics we want to keep track of
        self.precision = {model: {prompting_technique: 0 for prompting_technique in prompting_techniques} for model in models}
        self.recall = {model: {prompting_technique: 0 for prompting_technique in prompting_techniques} for model in models}
        self.f1 = {model: {prompting_technique: 0 for prompting_technique in prompting_techniques} for model in models}

    def evaluate(self, include_indices=False):
        """
        Evaluates the performance of the models.
        """
        for model in self.models:
            model = model()
            for prompting_technique in self.prompting_techniques:
                # Filler prompt to get the correct data
                prompt = prompting_technique(text="Filler", data=self.gold_standard, model=model)
                path = f"data/responses/{model.name}_{prompt.name}.jsonl"
                try:
                    data = Data(data_path=path)
                except FileNotFoundError:
                    print(f"Data for model: {model}, prompting technique: {prompting_technique} not found, skipping.")
                    continue
                
                for i in range(len(data)):
                    # G = Gold Standard labels, P = Predicted labels
                    _, P = data[i]
                    _, G = self.gold_standard[i]
                    precision = self.calculate_precision(P, G, include_indices)
                    recall = self.calculate_recall(P, G, include_indices)
                    f1 = self.calculate_f1_score(precision, recall)

                    # Save the metrics
                    self.precision[model][prompting_technique] += precision
                    self.recall[model][prompting_technique] += recall
                    self.f1[model][prompting_technique] += f1

    def plot(self, metric="f1"):
        """
        Plot the passed metric.
        """
        metric = getattr(self, metric)
        for model in self.models:
            plt.plot(self.prompting_techniques, [metric[model][prompting_technique] for prompting_technique in self.prompting_techniques], label=model)
        plt.legend()
        plt.show()


    def calculate_f1_score(self, precision, recall):
        """
        F1 score calculation (as the harmonic mean of precision and recall).
        """
        # Avoid division by zero
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_recall(self, P, G, include_indices):
        """
        Recall calculation using the same symbols as in the MAFALDA paper.

        Symbols:
        - P = Predicted labels
        - G = Gold Standard labels
        - G_minus = Gold Standard labels without the 'Nothing' label
        """
        G_minus = [g for g in G if 'Nothing' not in {label.name for label in g}]
        if not G_minus:
            return 1.0 # If there are no gold standard labels, recall is 1.0
        total_score = 0
        # Iterate over labels and sum the comparison scores
        for g_group in G_minus:
            max_score = 0
            for p_group in P:
                score = self._comparison(p_group, g_group, include_indices)
                max_score = max(max_score, score)
            total_score += max_score
        # We divide by the number of labels
        return total_score / len(G_minus)
    
    def calculate_precision(self, P, G, include_indices):
        """
        The precision function (as given in the MAFALDA paper).

        This of course is the same as the recall function, but with the roles of P and G reversed.
        And we do not exclude the 'Nothing' label, as only the nothing labels in the gold standard are excluded (because it is hard to expect a model to predict 'Nothing' labels for spans).
        """
        if not P:
            return 1.0
        total_score = 0
        for p_group in P:
            max_score = 0
            for g_group in G:
                score = self._comparison(p_group, g_group, include_indices)
                max_score = max(max_score, score)
            total_score += max_score
        return total_score / len(P)
    

    def _comparison(self, p_labels, g_labels, include_indices):
        """
        Comparison score (again as given in the MAFALDA paper).

        We want the option to either include or exclude indices in the comparison.
        If indices are included, we calculate the intersection of the indices of the predicted and gold standard labels.
        Otherwise, just compare label names.
        """
        if include_indices:
            p_indices = set.union(*(p.indices() for p in p_labels))
            g_indices = set.union(*(g.indices() for g in g_labels))
            intersection = len(p_indices.intersection(g_indices))
            # In the code of the MAFALDA paper, they give three options for h: PRED_SIZE, GOLD_SIZE, and JACCARD_INDEX.
            # We choose JACCARD_INDEX (as this is their default). Which they calculate as:
            # h = len(p_indices) + len(g_indices) - intersection
            p_length = sum(len(p.indices()) for p in p_labels)
            g_length = sum(len(g.indices()) for g in g_labels)
            if p_length + g_length - intersection == 0:
                return 0
            else:
                h = p_length + g_length - intersection
        else:
            intersection = 1 if self.delta([p.name for p in p_labels], [g.name for g in g_labels]) else 0
            h = 1

        delta = self.delta([p.name for p in p_labels], [g.name for g in g_labels])
        return (intersection / h) * delta

    def delta(labels1, labels2):
        """
        Similarity function (as given in the MAFALDA paper).
        """
        return int(any(label in labels2 for label in labels1))