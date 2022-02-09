import json
from Locator import CommMatching
from Rewriter import CommRewriting
import argparse
from datetime import datetime
import random
import numpy as np
import torch
from utils import load, feature_augmentation, split_communities, eval_scores
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write2file(comms, filename):
    with open(filename, 'w') as fh:
        content = '\n'.join([', '.join([str(i) for i in com]) for com in comms])
        fh.write(content)


def read4file(filename):
    with open(filename, "r") as file:
        pred = [[int(node) for node in x.split(', ')] for x in file.read().strip().split('\n')]
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # CommM related
    parser.add_argument("--conv_type", type=str, help="type of convolution", default="GCN")
    parser.add_argument("--n_layers", type=int, help="number of gnn layers", default=2)
    parser.add_argument("--hidden_dim", type=int, help="training hidden size", default=64)
    parser.add_argument("--output_dim", type=int, help="training hidden size", default=64)
    parser.add_argument("--dropout", type=float, help="dropout rate", default=0.2)
    parser.add_argument("--margin", type=float, help="margin loss", default=0.4)
    parser.add_argument("--fine_ratio", dest="fine_ratio", type=float, help="fine-grained sampling ratio", default=0.0)
    parser.add_argument("--comm_max_size", type=int, help="Community max size", default=12)

    # Train CommM
    parser.add_argument("--lr", dest="lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--device", dest="device", type=str, help="training device", default="cpu")
    parser.add_argument("--batch_size", type=int, help="training batch size", default=32)
    parser.add_argument("--pairs_size", type=int, help="pairs size", default=100)
    parser.add_argument("--seed", type=int, help="seed", default=0)

    parser.add_argument("--pred_size", type=int, help="pred size", default=1000)
    parser.add_argument("--commm_path", type=str, help="CommM path", default="")
    parser.add_argument("--commr_path", type=str, help="CommR path", default="")

    parser.add_argument("--dataset", type=str, help="dataset", default="amazon")

    # Train CommR
    parser.add_argument("--agent_lr", type=float, help="CommR learning rate", default=1e-3)
    parser.add_argument("--n_episode", type=int, help="number of episode", default=10)
    parser.add_argument("--n_epoch", type=int, help="number of epoch", default=1000)
    parser.add_argument("--gamma", type=float, help="CommR gamma", default=0.99)
    parser.add_argument("--max_step", type=int, help="", default=10)

    # Save log
    parser.add_argument("--writer_dir", type=str, help="Summary writer directory", default="")

    args = parser.parse_args()
    seed_all(args.seed)

    if not os.path.exists(f"ckpts/{args.dataset}"):
        os.mkdir(f"ckpts/{args.dataset}")
    args.writer_dir = f"ckpts/{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.comm_max_size = 20 if args.dataset.startswith("lj") else 12
    print(args.writer_dir)

    print('= ' * 20)
    print('##  Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    nodes, edges, communities = load(args.dataset)
    graph, _ = feature_augmentation(nodes, edges)
    train_comms, val_comms, test_comms = split_communities(communities, 90, 10)

    # Training CommMatching
    CommM_obj = CommMatching(args, graph, train_comms, val_comms)
    CommM_obj.train_epoch(1)
    pred_comms, feat_mat = CommM_obj.make_prediction()
    f, j, nmi = eval_scores(pred_comms, test_comms, tmp_print=True)
    metrics_string = '_'.join([f'{x:0.4f}' for x in [f, j, nmi]])
    write2file(pred_comms, args.writer_dir + "/CommM_" + metrics_string + '.txt')

    # Use F1-score as Reward function
    # Train CommRewriting
    cost_choice = "f1"
    # Note that feed predicted communities `pred_comms` into CommRewriting
    CommR_obj = CommRewriting(args, graph, feat_mat, train_comms, val_comms, pred_comms, cost_choice)
    CommR_obj.train()
    rewrite_comms = CommR_obj.get_rewrite()
    f, j, nmi = eval_scores(rewrite_comms, test_comms, tmp_print=True)
    metrics_string = '_'.join([f'{x:0.4f}' for x in [f, j, nmi]])
    write2file(rewrite_comms, args.writer_dir + f"/CommR_{cost_choice}_" + metrics_string + '.txt')

    # Save setting
    with open(args.writer_dir + '/settings.json', 'w') as fh:
        arg_dict = vars(args)
        json.dump(arg_dict, fh, sort_keys=True, indent=4)

    print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)
