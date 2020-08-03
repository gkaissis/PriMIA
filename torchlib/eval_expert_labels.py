import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from torchlib.utils import stats_table

if __name__ == "__main__":
    expert_labels = pd.read_csv("data/expert_labels.csv")
    el_list = expert_labels["Einschätzung"].to_list()

    labels = pd.read_csv("data/Labels.csv")
    labels = labels[labels["Dataset_type"] == "TEST"]
    labels = labels["Numeric_Label"].to_list()

    print(
        stats_table(
            confusion_matrix(labels, el_list),
            classification_report(labels, el_list, output_dict=True, zero_division=0),
            class_names=["normal", "bacterial", "viral"],
        )
    )

