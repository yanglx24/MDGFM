import torch
from torch_geometric.datasets import Planetoid
import numpy as np
from torch_geometric.datasets import (
    AttributedGraphDataset,
    Planetoid,
    Reddit,
    WikipediaNetwork,
    WebKB,
    AmazonProducts,
    FB15k_237,
    MoleculeNet,
    FacebookPagePage,
)
from ogb.nodeproppred import PygNodePropPredDataset


def gen_few_shot_data(dataset_name, num_shots, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="data", name=dataset_name)
    elif dataset_name in ["Chameleon", "Squirrel"]:
        dataset = WikipediaNetwork(root="data", name=dataset_name)
    elif dataset_name in ["Cornell"]:
        dataset = WebKB(root="data", name=dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data = dataset[0]

    labels = data.y
    unique_classes = torch.unique(labels)

    few_shot_indices = []
    few_shot_labels = []

    for c in unique_classes:
        class_indices = (labels == c).nonzero(as_tuple=True)[0]

        if len(class_indices) < num_shots:
            selected_indices = class_indices
        else:
            shuffled_indices = class_indices[torch.randperm(len(class_indices))]
            selected_indices = shuffled_indices[:num_shots]

        few_shot_indices.extend(selected_indices.tolist())
        few_shot_labels.extend(labels[selected_indices].tolist())

    return few_shot_indices, few_shot_labels

