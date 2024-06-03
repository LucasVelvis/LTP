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
        # TODO: Maybe fix this way of retrieving the names
        technique_names = [prompting_technique("Filler", "Filler", "Filler").name for prompting_technique in prompting_techniques]
        self.precision = {model.name: {technique: 0 for technique in technique_names} for model in models}
        self.recall = {model.name: {technique: 0 for technique in technique_names} for model in models}
        self.f1 = {model.name: {technique: 0 for technique in technique_names} for model in models}

    def evaluate(self, include_indices=False):
        """
        Evaluates the performance of the models.
        """
        for model in self.models:
            for prompting_technique in self.prompting_techniques:
                # Filler prompt to get the correct data
                prompt = prompting_technique(text="Filler", data=self.gold_standard, model=model)
                print(f"Evaluating model: {model.name} for prompting technique: {prompt.name}")
                path = f"data/responses/{model.name}_{prompt.name}.jsonl"

                try:
                    data = Data(data_path=path)
                except FileNotFoundError:
                    print(f"Data for model: {model}, prompting technique: {prompting_technique} not found, skipping.")
                    continue

                # G = Gold Standard labels, P = Predicted labels
                P = [label for _, label in data]
                G = [label for _, label in self.gold_standard]

                # Fix uppercase error in "Nothing" of the predicted labels
                for p in P:
                    for label in p:
                        if label.name == "Nothing":
                            label.name = "nothing"
                    # Remove exact duplicates from the predicted labels
                    p = set(p)

                precision = self.calculate_precision(P, G, include_indices)
                recall = self.calculate_recall(P, G, include_indices)
                f1 = self.calculate_f1_score(precision, recall)

                print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

                # Save the metrics
                self.precision[model.name][prompt.name] = precision
                self.recall[model.name][prompt.name] = recall
                self.f1[model.name][prompt.name] = f1

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
        total_scores = []

        for P, G in zip(P, G):
            G_minus = [g for g in G if g.name.lower() != "nothing"]
            if not G_minus:
                total_scores.append(1.0)  # If there are no gold standard labels, recall is 1.0
                continue

            recall_scores = []
            for g in G_minus:
                max_score = 0
                for p in P:
                    score = self._comparison(p, g, include_indices)
                    max_score = max(max_score, score)
                recall_scores.append(max_score)
            total_scores.append(sum(recall_scores) / len(G_minus))
        return sum(total_scores) / len(total_scores)

    def calculate_precision(self, P, G, include_indices):
        """
        The precision function (as given in the MAFALDA paper).

        This of course is the same as the recall function, but with the roles of P and G reversed.
        And we do not exclude the 'Nothing' label, as only the nothing labels in the gold standard are excluded (because it is hard to expect a model to predict 'Nothing' labels for spans).
        """
        total_scores = []
        for P, G in zip(P, G):
            if not P:
                total_scores.append(0)  # If there are no predicted labels, precision is 0
                continue
            
            precision_scores = []
            for p in P:
                max_score = 0
                for g in G:
                    score = self._comparison(p, g, include_indices)
                    max_score = max(max_score, score)
                precision_scores.append(max_score)                
            total_scores.append(sum(precision_scores) / len(P))
        return sum(total_scores) / len(total_scores)


    def _comparison(self, p, g, include_indices):
        """
        Comparison score (again as given in the MAFALDA paper).

        We want the option to either include or exclude indices in the comparison.
        If indices are included, we calculate the intersection of the indices of the predicted and gold standard labels.
        Otherwise, just compare label names.
        """
        if include_indices:
            intersection = len(p.indices().intersection(g.indices()))
            # In the code of the MAFALDA paper, they give three options for h (as defined in the paper): PRED_SIZE, GOLD_SIZE, and JACCARD_INDEX.
            # We choose JACCARD_INDEX (as this is their default). Which they calculate as:
            # h = len(p_indices) + len(g_indices) - intersection
            # This is the same as the Jaccard index, which is the intersection divided by the union.
            h = len(p.indices()) + len(g.indices()) - intersection
        else:
            # If we do not include indices, we just compare the label names, 
            intersection = self.delta(p, g)  # Intersection is 1 if the names are the same, 0 otherwise.
            h = 1

        # Compare the labels
        delta = self.delta(p, g)
        return (intersection / h) * delta

    def delta(self, label1, label2):
        """
        Similarity function (as given in the MAFALDA paper).
        """
        return label1.name.lower() == label2.name.lower()

    def plot(self):
        """
        Make a bar plot of the evaluation metrics.
        With a subplot for each metric.
        """
        # We will use pandas, because it was very hard to get the bars to not
        # Overlap with matplotlib.
        import pandas as pd

        precision_df = pd.DataFrame(self.precision)
        recall_df = pd.DataFrame(self.recall)
        f1_df = pd.DataFrame(self.f1)

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        colors = ["#003f5c", "#bc5090", "#ffa600"]

        precision_df.plot(kind="bar", ax=axs[0], title="Precision", color=colors)
        recall_df.plot(kind="bar", ax=axs[1], title="Recall", color=colors)
        f1_df.plot(kind="bar", ax=axs[2], title="F1", color=colors)

        for i, ax in enumerate(axs):
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)  # Plot full metric range
            ax.legend(loc="upper right")  # Legend was in the way
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # X-labels were rotated

            for container in ax.containers:  # Include the values on the bars
                # Rounded to 2 decimals
                ax.bar_label(container, fmt="%.2f", label_type="edge")

        plt.tight_layout()
        # plt.show()
        plt.savefig("data/evaluation.png", dpi=300)
