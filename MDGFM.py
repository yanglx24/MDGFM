import torch
import torch.nn as nn
from utils.data_process import KGNodeInitializer
import numpy as np
import scipy.sparse as sp
import random
import time
import math
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.datasets import (
    AttributedGraphDataset,
    Planetoid,
    Amazon,
    Reddit,
    WikipediaNetwork,
    WebKB,
    AmazonProducts,
    FB15k_237,
    WordNet18RR,
    MoleculeNet,
)
from ogb.nodeproppred import PygNodePropPredDataset
import preprompt
from preprompt import PrePrompt, pca_compression
from utils import process, logger
import os
import argparse
from downprompt import downprompt, prefeatureprompt
import csv
from tqdm import tqdm 

def parse_args():
    parser = argparse.ArgumentParser("MDGFM")

    parser.add_argument("--dataset", type=str, default="Chameleon", help="data")
    parser.add_argument("--drop_percent", type=float, default=0.1, help="drop percent")
    parser.add_argument("--lr", type=float, default=0.02, help="pretrain lr")
    parser.add_argument("--downstreamlr", type=float, default=0.003, help="downstream lr")
    parser.add_argument("--epochs", type=int, default=60, help="epoch")
    parser.add_argument("--shot_num", type=int, default=1, help="shotnum")
    parser.add_argument("--seed", type=int, default=1024, help="seed")
    parser.add_argument(
        "--save_name",
        type=str,
        default="pretrain_on_6_datasets.pt",
        help="save ckpt name",
    )
    parser.add_argument(
        "--combinetype", type=str, default="mul", help="the type of text combining"
    )
    return parser.parse_args()

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_KG_data_with_feats(dataset_name):
    assert dataset_name in [
        "FB15k_237",
        "WordNet18RR",
    ], "Only FB15k_237 and WordNet18RR are supported."
    if dataset_name == "FB15k_237":
        dataset = FB15k_237("data/FB15k_237", split="train")
    elif dataset_name == "WordNet18RR":
        dataset = WordNet18RR("data/WordNet18RR", split="train")
    if os.path.exists(f"data/{dataset_name}/results.pt"):
        results = torch.load(
            f"data/{dataset_name}/results.pt",
            map_location=torch.device("cpu"),
        )
        dataset.data.x = results["node_embeddings"]
        return dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_name == "FB15k_237":
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
        dataset.x = results["node_embeddings"]
    # elif dataset_name == "WordNet18RR":
    #     train_data = WordNet18RR("data/WordNet18RR")[0]
    #     valid_data = WordNet18RR("data/WordNet18RR", split="valid")[0]
    #     test_data = WordNet18RR("data/WordNet18RR", split="test")[0]
    #     model = KGNodeInitializer("transe", device=device)
    #     results = model.fit(
    #         train_data,
    #         valid_data,
    #         test_data,
    #         batch_size=1000,
    #         epochs=500,
    #         verbose=True,
    #     )
    #     data = WordNet18RR("data/WordNet18RR", split="train")[0]
    #     data.x = results["node_embeddings"]
    torch.save(results, f"data/{dataset_name}/results.pt")

    return dataset

def load_dataset(TARGET_NUM_BATCHES=200):
    dataset1 = PygNodePropPredDataset(root="data", name="ogbn-arxiv")
    dataset2 = AmazonProducts(root="data/AmazonProducts")
    dataset3 = Reddit(root="data/Reddit")
    dataset4 = load_KG_data_with_feats("FB15k_237")
    dataset5 = AttributedGraphDataset(root="data", name="PPI")
    dataset6 = MoleculeNet(root="data", name="PCBA")

    datasets = {
        "ogbn-arxiv": {"data": dataset1, "type": "single"},
        "AmazonProducts": {"data": dataset2, "type": "single"},
        "Reddit": {"data": dataset3, "type": "single"},
        "FB15k_237": {"data": dataset4, "type": "single"},
        "PPI": {"data": dataset5, "type": "single"},
        "PCBA": {"data": dataset6, "type": "multi"} 
    }

    batch_sizes = {}

    for name, info in datasets.items():
        if info['type'] == 'single':
            num_items = info['data'].data.num_nodes
        else:
            num_items = len(info['data'])

        bs = math.ceil(num_items / TARGET_NUM_BATCHES)
        batch_sizes[name] = bs
        
    num_neighbors_config = [5, 6]

    loader1 = NeighborLoader(dataset1.data, batch_size=batch_sizes["ogbn-arxiv"], num_neighbors=num_neighbors_config, shuffle=True)
    loader2 = NeighborLoader(dataset2.data, batch_size=batch_sizes["AmazonProducts"], num_neighbors=num_neighbors_config, shuffle=True)
    loader3 = NeighborLoader(dataset3.data, batch_size=batch_sizes["Reddit"], num_neighbors=num_neighbors_config, shuffle=True)
    loader4 = NeighborLoader(dataset4.data, batch_size=batch_sizes["FB15k_237"], num_neighbors=num_neighbors_config, shuffle=True)
    loader5 = NeighborLoader(dataset5.data, batch_size=batch_sizes["PPI"], num_neighbors=num_neighbors_config, shuffle=True)
    loader6 = DataLoader(dataset6, batch_size=batch_sizes["PCBA"], shuffle=True)
    return loader1, loader2, loader3, loader4, loader5, loader6

logger = logger.Logger(log_dir="logs").get_logger()
args = parse_args()
print(dir(logger))
seed(args.seed)
loader1, loader2, loader3, loader4, loader5, loader6 = load_dataset(TARGET_NUM_BATCHES=1000)

patience = 10
l2_coef = 0.0001
hid_units = 256
sparse = True
LP = False
testnum = 10000
downstreamlr = args.downstreamlr
dataset = args.dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnt_wait = 0
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
unify_dim = 5
best = 1e9
args.save_name = str(time.localtime()) + args.save_name

model = PrePrompt(unify_dim, hid_units, 3, args.drop_percent, args.combinetype).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2_coef)

logger.info("===================== Start Pre-training on 6 datasets... =====================")
model.train()
for epoch in range(args.epochs):
    epoch_total_loss = 0
    for batch, (data1, data2, data3, data4, data5, data6) in enumerate(
        zip(loader1, loader2, loader3, loader4, loader5, loader6)
    ):
        logger.info("Batch:", batch) 
        results = [
            process.process_tu(data, data.x.shape[1]) 
            for data in [data1, data2, data3, data4, data5, data6]
        ]
        all_features, all_adjs = zip(*results)
        features11, features22, features33, features44, features55, features66 = all_features
        adj1, adj2, adj3, adj4, adj5, adj6 = all_adjs
        pre_i = 6
        # if args.dataset == "Cora":
        #     features11, adj1 = process.process_tu(data6, data6.x.shape[1])
        #     pre_i = 0
        # elif args.dataset == "Pubmed":
        #     features22, adj2 = process.process_tu(data6, data6.x.shape[1])
        #     pre_i = 1
        # elif args.dataset == "Citeseer":
        #     features33, adj3 = process.process_tu(data6, data6.x.shape[1])
        #     pre_i = 2
        # elif args.dataset == "Chameleon":
        #     features44, adj4 = process.process_tu(data6, data6.x.shape[1])
        #     pre_i = 3
        # elif args.dataset == "Squirrel":
        #     features55, adj5 = process.process_tu(data6, data6.x.shape[1])
        #     pre_i = 4

        features1, features2, features3, features4, features5, features6 = [
            torch.FloatTensor(pca_compression(features, k=unify_dim)).to(device)
            for features in [features11, features22, features33, features44, features55, features66]
        ]

        adj = process.combine_dataset(adj1, adj2, adj3, adj4, adj5, adj6)

        adj1, adj2, adj3, adj4, adj5, adj6 = [
            process.normalize_adj(adj + sp.eye(adj.shape[0]))
            for adj in [adj1, adj2, adj3, adj4, adj5, adj6]
        ]

        if sparse:
            sp_adj1, sp_adj2, sp_adj3, sp_adj4, sp_adj5, sp_adj6 = [
                process.sparse_mx_to_torch_sparse_tensor(adj).to(device)
                for adj in [adj1, adj2, adj3, adj4, adj5, adj6]
            ]
        
        optimizer.zero_grad()
        kwargs = {
            "seqs": [features1, features2, features3, features4, features5, features6],
            "adjs": [sp_adj1, sp_adj2, sp_adj3, sp_adj4, sp_adj5, sp_adj6] if sparse else [adj1, adj2, adj3, adj4, adj5, adj6],
        }
        
        loss = model(sparse, pre_i, **kwargs)
        epoch_total_loss += loss.item()
        loss.backward()
        # found_nan_grad = False
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #             print(f"!!! Found NaN/Inf gradient in parameter: {name} !!!")
        #             found_nan_grad = True

        # if found_nan_grad:
        #     print("Stopping training due to NaN gradients.")
        #     import sys
        #     sys.exit()
        optimizer.step()
        print(f"Loss:[{loss.item():.4f}]")
    if epoch_total_loss < best:
        best = epoch_total_loss
        best_epoch = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), args.save_name)
    else:
        cnt_wait += 1
    if cnt_wait == patience:
        print("Early stopping!")
        break
        # print(f"Loading {best_t}th epoch")

# model = PrePrompt(unify_dim, hid_units, 1, 3, 0.1, args.combinetype)

# print("#" * 50)
# print("Downastream dataset is ", args.dataset)

# if args.dataset == "Cora" or args.dataset == "Citeseer" or args.dataset == "Pubmed":
#     dataset = Planetoid(root="data", name=args.dataset)
#     downk = 30
#     if args.dataset == "Pubmed":
#         testnum = 100
# if args.dataset == "Chameleon" or args.dataset == "Squirrel":
#     dataset = WikipediaNetwork(root="data", name=args.dataset)
#     downk = 15
# if args.dataset == "Cornell":
#     dataset = WebKB(root="data", name=args.dataset)
#     downk = 15

# print(args.dataset)
# loader = DataLoader(dataset)
# for data in loader:
#     print(data)
#     features, adj = process.process_tu(data, data.x.shape[1])
#     features = pca_compression(features, k=unify_dim)
#     adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
#     sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
#     sp_adj = sp_adj.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     features = torch.FloatTensor(features).to(
#         torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     )
#     print(features.shape)
#     ln = data.y.shape[0] - testnum
#     if ln < 0:
#         ln = 0
#     idx_test = range(ln, data.y.shape[0])
#     labels = data.y
#     data = np.array(data.y)
#     np.unique(data)
#     nb_classes = len(np.unique(data))
#     print(nb_classes)

# model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# model.load_state_dict(torch.load(args.save_name))

# embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None, LP)
# acclist = torch.FloatTensor(100).to(
#     torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )


# print(labels.shape)
# test_lbls = labels[idx_test].to(
#     torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )
# tot = torch.zeros(1)
# tot = tot.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# accs = []
# print("-" * 100)
# for shotnum in range(args.shot_num, args.shot_num + 1):
#     tot = torch.zeros(1)
#     tot = tot.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     accs = []
#     cnt_wait = 0
#     best = 1e9
#     best_t = 0
#     print("shotnum", shotnum)
#     for i in tqdm(range(50)):
#         log = downprompt(
#             model.texttoken1.weight.detach(),
#             model.texttoken2.weight.detach(),
#             model.texttoken3.weight.detach(),
#             model.texttoken4.weight.detach(),
#             model.texttoken5.weight.detach(),
#             model.sumtext.weight.detach(),
#             model.pretext1.weight.detach(),
#             model.pretext2.weight.detach(),
#             model.pretext3.weight.detach(),
#             model.pretext4.weight.detach(),
#             model.pretext5.weight.detach(),
#             model.balancetoken1.weight.detach(),
#             model.balancetoken2.weight.detach(),
#             model.balancetoken3.weight.detach(),
#             model.balancetoken4.weight.detach(),
#             model.balancetoken5.weight.detach(),
#             hid_units,
#             nb_classes,
#             args.combinetype,
#             unify_dim,
#         ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         idx_train, train_lbls = gen_few_shot_data(
#             args.dataset, shotnum, seed + i
#         )
#         idx_train = (
#             torch.tensor(idx_train)
#             .type(torch.long)
#             .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         )
#         pretrain_embs = embeds[0, idx_train]
#         test_embs = embeds[0, idx_test]
#         train_lbls = (
#             torch.tensor(train_lbls)
#             .type(torch.long)
#             .squeeze()
#             .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         )
#         opt = torch.optim.Adam([{"params": log.parameters()}], lr=args.downstreamlr)
#         log = log.to(
#             torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         )
#         best = 1e9
#         pat_steps = 0
#         best_acc = torch.zeros(1)
#         best_acc = best_acc.to(
#             torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         )

#         for _ in range(400):
#             log.train()
#             opt.zero_grad()
#             logits = (
#                 log(
#                     features,
#                     sp_adj,
#                     sparse,
#                     model.gcn,
#                     idx_train,
#                     pretrain_embs,
#                     downk,
#                     train_lbls,
#                     1,
#                 )
#                 .float()
#                 .to(
#                     torch.device("cuda" if torch.cuda.is_available() else "cpu")
#                 )
#             )
#             loss = xent(logits, train_lbls)
#             if loss < best:
#                 best = loss
#                 cnt_wait = 0
#             else:
#                 cnt_wait += 1
#             if cnt_wait == patience:
#                 print("Early stopping!")
#                 break

#             loss.backward(retain_graph=True)
#             opt.step()
#         logits = log(
#             features, sp_adj, sparse, model.gcn, idx_test, test_embs, downk
#         )
#         preds = torch.argmax(logits, dim=1).to(
#             torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         )
#         acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
#         accs.append(acc * 100)
#         print("acc:[{:.4f}]".format(acc))
#         tot += acc
#     print("-" * 100)
#     print("Average accuracy:[{:.4f}]".format(tot.item() / 50))
#     accs = torch.stack(accs)
#     print("Mean:[{:.4f}]".format(accs.mean().item()))
#     print("Std :[{:.4f}]".format(accs.std().item()))
#     print("-" * 100)
#     row = [
#         "Final:",
#         "lr",
#         args.lr,
#         "downstreamlr",
#         args.downstreamlr,
#         "epochs",
#         args.epochs,
#         hid_units,
#         accs.mean().item(),
#         accs.std().item(),
#     ]
#     out = open(
#         "data/ICML25_{}_fewshot.csv".format(args.dataset.lower()),
#         "a",
#         newline="",
#     )
#     csv_writer = csv.writer(out, dialect="excel")
#     csv_writer.writerow(row)
