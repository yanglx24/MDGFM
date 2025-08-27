import torch
from torch_geometric.datasets import Planetoid
import numpy as np
from data_process import KGNodeInitializer
from torch_geometric.datasets import (
    AttributedGraphDataset,
    TUDataset,
    Planetoid,
    Amazon,
    Reddit,
    WikipediaNetwork,
    WebKB,
    AmazonProducts,
    FB15k_237,
    WordNet18RR,
    MoleculeNet,
    FacebookPagePage,
)
from ogb.nodeproppred import PygNodePropPredDataset


def load_FB15k237_data_with_feats():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = FB15k_237("data/FB15k_237", split="train")[0]
    valid_data = FB15k_237("data/FB15k_237", split="val")[0]
    test_data = FB15k_237("data/FB15k_237", split="test")[0]
    model = KGNodeInitializer("transe", device=device)
    results = model.fit(
        train_data,
        valid_data,
        test_data,
        batch_size=1000,
        epochs=500,
        verbose=True,
    )
    data = FB15k_237("data/FB15k_237", split="train")[0]
    data.x = results["node_embeddings"]
    return data


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


def get_dataset_stats(dataset_name):
    if dataset_name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root="data", name=dataset_name)
    elif dataset_name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root="data", name=dataset_name)
    elif dataset_name in ["cornell"]:
        dataset = WebKB(root="data", name=dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    data = dataset[0]

    stats = {
        "dataset_name": dataset_name,
        "num_graphs": len(dataset),
        "num_nodes": data.num_nodes,
        "num_classes": dataset.num_classes,
    }

    return stats


if __name__ == "__main__":
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="data")
    dataset = AmazonProducts(root="data/AmazonProducts")
    dataset = Reddit(root="data/Reddit")
    dataset = FB15k_237(root="data/FB15k_237")
    dataset = AttributedGraphDataset(root="data", name="PPI")
    print(dataset[0].data)
    dataset = MoleculeNet(root="data", name="PCBA")

    # dataset = Planetoid(root="data", name="Pubmed")
    # dataset = Amazon(root="data", name="Computers")
    dataset = FacebookPagePage(root="data/FacebookPagePage")
    # dataset = WordNet18RR(root="data/WordNet18RR")
    # dataset = TUDataset(root="data", name="PROTEINS")
    # dataset = MoleculeNet(root="data", name="Tox21")

    # datasets = ["Cora", "Citeseer", "Pubmed", "Chameleon", "Squirrel", "Cornell"]
    # num_shots = 5
    # seed = 41
    # for dataset_name in datasets:
    #     stats = get_dataset_stats(dataset_name)
    #     print(f"Dataset: {stats['dataset_name']}")
    #     print(f"Number of graphs: {stats['num_graphs']}")
    #     print(f"Number of nodes: {stats['num_nodes']}")
    #     print(f"Number of classes: {stats['num_classes']}")

    #     print(f"Generating few-shot data for {dataset_name} with {num_shots} shots")
    #     few_shot_indices, few_shot_labels = gen_few_shot_data(
    #         dataset_name, num_shots, seed
    #     )
    #     print(f"Few-shot indices: {few_shot_indices}")
    #     print(len(few_shot_indices), "few-shot samples generated.")
    #     print(f"Few-shot labels: {few_shot_labels}")
    #     print("=========================================================")
