from data import Data
from typing import List
from util import get_all_models, get_all_prompting_techniques
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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

        # Attempt at some new metrics
        self.confusion_matrices = {model.name: {technique: None for technique in technique_names} for model in models}

    def evaluate(self, include_indices=True):
        """
        Evaluates the performance of the models.
        """
        self.include_indices = include_indices
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
                
                # Sort both data such that they are in the same order
                data = sorted(data, key=lambda x: x[0])
                gold_standard = sorted(self.gold_standard, key=lambda x: x[0])

                # G = Gold Standard labels, P = Predicted labels
                P = [label for _, label in data]
                G = [label for _, label in gold_standard]

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

                # Save the f1 metrics
                self.precision[model.name][prompt.name] = precision
                self.recall[model.name][prompt.name] = recall
                self.f1[model.name][prompt.name] = f1

                # -- Extra metrics --
                # Compute and store confusion matrix
                cm, classes = self.compute_confusion_matrix(P, G)
                self.confusion_matrices[model.name][prompt.name] = {'matrix': cm, 'classes': classes}

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
        if self.include_indices:
            plt.savefig("data/evaluation_spans.png", dpi=300)
        else:
            plt.savefig("data/evaluation_simple.png", dpi=300)
        # plt.show()
        self.plot_combined_confusion_matrices_all()
        self.plot_combined_confusion_matrices_models_per_technique()


## ------------------------------------------------
## Some extra metrics
## NB: These metrics were created with the help of Github Copilot (AI pair programming tool) due to time constraints and the complexity of the task.
## ------------------------------------------------

    def compute_confusion_matrix(self, P, G):
        """
        Compute the confusion matrix for each class.
        """
        classes = set()
        for g_list in G:
            for g in g_list:
                classes.add(g.name.lower())
        for p_list in P:
            for p in p_list:
                classes.add(p.name.lower())
        
        class_to_index = {cls: i for i, cls in enumerate(classes)}
        
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        
        for p_list, g_list in zip(P, G):
            p_indices = [class_to_index[p.name.lower()] for p in p_list]
            g_indices = [class_to_index[g.name.lower()] for g in g_list]
            
            for p_idx in p_indices:
                for g_idx in g_indices:
                    cm[g_idx][p_idx] += 1
        
        return cm, list(classes)
    
    def expand_confusion_matrix(self, cm, classes, all_classes):
        """
        Expand the confusion matrix to include all classes, setting missing classes to zero.
        """
        expanded_cm = np.zeros((len(all_classes), len(all_classes)), dtype=int)
        class_to_index = {cls: i for i, cls in enumerate(all_classes)}
        
        for i, cls1 in enumerate(classes):
            for j, cls2 in enumerate(classes):
                expanded_cm[class_to_index[cls1]][class_to_index[cls2]] = cm[i][j]
        
        return expanded_cm

    def plot_combined_confusion_matrices_models_per_technique(self):
        """
        Plot combined confusion matrices for all models per technique.
        """
        all_classes = sorted({cls for model_metrics in self.confusion_matrices.values() for cm_data in model_metrics.values() for cls in cm_data['classes']})

        for prompting_technique in self.prompting_techniques:
            technique_name = prompting_technique("Filler", "Filler", "Filler").name
            combined_cm = np.zeros((len(all_classes), len(all_classes)), dtype=float)
            model_count = 0

            for model_name, model_metrics in self.confusion_matrices.items():
                if technique_name in model_metrics:
                    cm_data = model_metrics[technique_name]
                    cm = self.expand_confusion_matrix(cm_data['matrix'], cm_data['classes'], all_classes)
                    combined_cm += cm
                    model_count += 1

            if model_count > 0:
                combined_cm = combined_cm / model_count  # Normalize by the number of models

            fig, ax = plt.subplots(figsize=(25, 15))
            sns.heatmap(combined_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes, ax=ax)
            ax.set_title(f'Combined Confusion Matrix for {technique_name} (Averaged over Models)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

            plt.tight_layout()
            plt.savefig(f"data/confusion_matrices/combined_confusion_matrix_models_{technique_name}.png", dpi=300)
            # plt.show()

    def plot_combined_confusion_matrices_all(self):
        """
        Plot combined confusion matrices for all models and techniques.
        """
        all_classes = sorted({cls for model_metrics in self.confusion_matrices.values() for cm_data in model_metrics.values() for cls in cm_data['classes']})

        combined_cm = np.zeros((len(all_classes), len(all_classes)), dtype=float)
        model_count = 0

        for model_name, model_metrics in self.confusion_matrices.items():
            for technique_name, cm_data in model_metrics.items():
                cm = self.expand_confusion_matrix(cm_data['matrix'], cm_data['classes'], all_classes)
                combined_cm += cm
                model_count += 1

        if model_count > 0:
            combined_cm = combined_cm / model_count

        fig, ax = plt.subplots(figsize=(25, 15))
        sns.heatmap(combined_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes, ax=ax)
        ax.set_title('Combined Confusion Matrix for All Models and Techniques (Averaged)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        plt.tight_layout()
        plt.savefig("data/confusion_matrices/combined_confusion_matrix_all.png", dpi=300)
        # plt.show()

    
