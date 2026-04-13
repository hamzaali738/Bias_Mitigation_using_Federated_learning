"""
Federated Learning with Bias Mitigation — 3 Experiment Modes
=============================================================
Usage:
  python train_federated.py --mode baseline      [--num_clients 5] ...
  python train_federated.py --mode lff_avg        [--num_clients 5] ...
  python train_federated.py --mode lff_weighted   [--num_clients 5] ...
"""

import os, sys, copy, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict

# Project imports
from data.util import get_dataset, IdxDataset, CMNISTDataset, create_fl_validation_set
from module.loss import GeneralizedCELoss
from module.util import get_backbone
from util import EMA


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def fedavg_aggregate(state_dicts, weights):
    """Weighted average of model state dicts."""
    w = [wi / sum(weights) for wi in weights]
    agg = {}
    for key in state_dicts[0]:
        agg[key] = sum(sd[key].float() * wi for sd, wi in zip(state_dicts, w))
    return agg


def evaluate_model(model, data_loader, device, target_attr_idx=0, bias_attr_idx=1):
    """Evaluate a single model; returns total_acc, conflict_acc."""
    model.eval()
    correct_total, total = 0, 0
    correct_conflict, n_conflict = 0, 0
    with torch.no_grad():
        for data, attr, _ in data_loader:
            data = data.to(device)
            target = attr[:, target_attr_idx].to(device)
            bias   = attr[:, bias_attr_idx].to(device)
            logit  = model(data)
            pred   = logit.argmax(dim=1)
            correct = (pred == target)
            is_conflict = (target != bias)

            correct_total += correct.sum().item()
            total += data.size(0)
            correct_conflict += (correct & is_conflict).sum().item()
            n_conflict += is_conflict.sum().item()
    model.train()
    acc_total = correct_total / max(total, 1) * 100
    acc_conflict = correct_conflict / max(n_conflict, 1) * 100
    return acc_total, acc_conflict


# ═══════════════════════════════════════════════════════════
#  Local Training Functions
# ═══════════════════════════════════════════════════════════

def train_baseline_local(model_d, train_loader, optimizer, criterion, device,
                         num_epochs, target_attr_idx=0):
    """Standard CE training on a single debiased model (no bias model)."""
    model_d.train()
    for ep in range(num_epochs):
        for data, attr, _ in train_loader:
            data  = data.to(device)
            label = attr[:, target_attr_idx].to(device)
            logit = model_d(data)
            loss  = criterion(logit, label).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_lff_local(model_b, model_d, train_loader, optimizer_b, optimizer_d,
                    criterion, bias_criterion, sample_loss_ema_b,
                    sample_loss_ema_d, num_classes, device,
                    num_epochs, target_attr_idx=0, bias_attr_idx=1):
    """LfF local training: dual model with re-weighting (from lff_production.py)."""
    model_b.train()
    model_d.train()
    for ep in range(num_epochs):
        for index, data, attr, _ in train_loader:
            data  = data.to(device)
            attr  = attr.to(device)
            index = index.to(device)
            label      = attr[:, target_attr_idx]
            bias_label = attr[:, bias_attr_idx]

            logit_b = model_b(data)
            logit_d = model_d(data)

            loss_b = criterion(logit_b, label).cpu().detach().to(device)
            loss_d = criterion(logit_d, label).cpu().detach().to(device)

            # EMA update
            sample_loss_ema_b.update(loss_b, index)
            sample_loss_ema_d.update(loss_d, index)

            # Class-wise normalise
            loss_b = sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = sample_loss_ema_d.parameter[index].clone().detach()
            label_cpu = label.cpu()
            for c in range(num_classes):
                ci = np.where(label_cpu == c)[0]
                max_b = sample_loss_ema_b.max_loss(c) + 1e-8
                max_d = sample_loss_ema_d.max_loss(c)
                loss_b[ci] /= max_b
                loss_d[ci] /= max_d

            loss_weight = loss_b / (loss_b + loss_d + 1e-8)

            loss_b_update = bias_criterion(logit_b, label)
            loss_d_update = criterion(logit_d, label) * loss_weight.to(device)
            loss = loss_b_update.mean() + loss_d_update.mean()

            optimizer_b.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_b.step()
            optimizer_d.step()


# ═══════════════════════════════════════════════════════════
#  Main FL Loop
# ═══════════════════════════════════════════════════════════

def run_federated(args):
    device = torch.device(args.device)
    num_classes = 10  # CMNIST

    # ---- 1. Load full CMNIST training set (raw, un-wrapped) ----
    data2preprocess = {'cmnist': None}
    train_dataset_raw = get_dataset(
        args.dataset, data_dir=args.data_dir, dataset_split="train",
        transform_split="train", percent=args.percent,
        use_preprocess=data2preprocess.get(args.dataset)
    )

    # ---- 2. Carve 200-sample validation set ----
    val_indices, train_indices = create_fl_validation_set(train_dataset_raw, num_per_class=10)
    val_subset   = Subset(train_dataset_raw, val_indices)
    train_subset = Subset(train_dataset_raw, train_indices)

    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # ---- 3. Partition training data IID across clients ----
    n = len(train_indices)
    perm = np.random.permutation(n).tolist()
    chunk = n // args.num_clients
    client_subsets = []
    for c in range(args.num_clients):
        start = c * chunk
        end   = n if c == args.num_clients - 1 else (c + 1) * chunk
        local_idx = [train_indices[perm[i]] for i in range(start, end)]
        client_subsets.append(Subset(train_dataset_raw, local_idx))
    print(f"[FL] {args.num_clients} clients, sizes: "
          f"{[len(cs) for cs in client_subsets]}")

    # ---- 4. Load test set ----
    test_dataset = get_dataset(
        args.dataset, data_dir=args.data_dir, dataset_split="test",
        transform_split="valid", percent=args.percent,
        use_preprocess=data2preprocess.get(args.dataset)
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # ---- 5. Initialise global models ----
    global_model_d = get_backbone(args.model, num_classes, args=args,
                                  pretrained=False).to(device)
    global_model_b = None
    if args.mode != 'baseline':
        global_model_b = get_backbone(args.model, num_classes, args=args,
                                      pretrained=False).to(device)

    criterion      = nn.CrossEntropyLoss(reduction='none')
    bias_criterion = GeneralizedCELoss(q=args.q)

    # ---- 6. FL rounds ----
    for rnd in range(1, args.num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  FL Round {rnd}/{args.num_rounds}  |  mode={args.mode}")
        print(f"{'='*60}")

        client_states_d = []
        client_states_b = []
        client_data_sizes = []
        client_val_accs = []

        for cid in range(args.num_clients):
            # -- clone global model(s) for this client --
            local_d = copy.deepcopy(global_model_d)
            opt_d   = optim.Adam(local_d.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

            if args.mode == 'baseline':
                # ---- BASELINE: plain CE, single model ----
                loader = DataLoader(client_subsets[cid],
                                    batch_size=args.batch_size,
                                    shuffle=True, drop_last=True,
                                    num_workers=args.num_workers)
                train_baseline_local(local_d, loader, opt_d, criterion,
                                     device, args.local_epochs)
            else:
                # ---- LfF modes: dual model ----
                local_b = copy.deepcopy(global_model_b)
                opt_b   = optim.Adam(local_b.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

                # Wrap client subset with IdxDataset for index tracking
                idx_subset = IdxDataset(client_subsets[cid])
                loader = DataLoader(idx_subset,
                                    batch_size=args.batch_size,
                                    shuffle=True, drop_last=True,
                                    num_workers=args.num_workers)

                # Build per-client EMA trackers
                client_labels = []
                for i in range(len(client_subsets[cid])):
                    _, attr, _ = client_subsets[cid][i]
                    client_labels.append(int(attr[0]))
                client_labels_t = torch.LongTensor(client_labels)

                ema_b = EMA(client_labels_t, num_classes=num_classes,
                            alpha=args.ema_alpha)
                ema_d = EMA(client_labels_t, num_classes=num_classes,
                            alpha=args.ema_alpha)

                train_lff_local(local_b, local_d, loader, opt_b, opt_d,
                                criterion, bias_criterion, ema_b, ema_d,
                                num_classes, device, args.local_epochs)
                client_states_b.append(copy.deepcopy(local_b.state_dict()))

            client_states_d.append(copy.deepcopy(local_d.state_dict()))
            client_data_sizes.append(len(client_subsets[cid]))

            # -- For weighted mode: evaluate client model on val set --
            if args.mode == 'lff_weighted':
                vacc, _ = evaluate_model(local_d, val_loader, device)
                client_val_accs.append(vacc)
                print(f"  Client {cid} val acc: {vacc:.2f}%")

        # ---- Aggregation ----
        if args.mode == 'lff_weighted':
            # Accuracy-proportional weights
            total_acc = sum(client_val_accs) + 1e-8
            weights = [a / total_acc for a in client_val_accs]
            print(f"  Aggregation weights (acc): "
                  f"{[f'{w:.3f}' for w in weights]}")
        else:
            # Standard FedAvg: data-size proportional
            weights = [float(s) for s in client_data_sizes]

        global_model_d.load_state_dict(
            fedavg_aggregate(client_states_d, weights))
        if args.mode != 'baseline':
            global_model_b.load_state_dict(
                fedavg_aggregate(client_states_b, weights))

        # ---- Evaluate global model after this round ----
        acc_total, acc_conflict = evaluate_model(
            global_model_d, test_loader, device)
        print(f"  >> Round {rnd} Test Acc: {acc_total:.2f}%  |  "
              f"Conflict Acc: {acc_conflict:.2f}%")

    # ---- 7. Final results ----
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  —  mode={args.mode}")
    print(f"{'='*60}")
    final_acc, final_conflict = evaluate_model(
        global_model_d, test_loader, device)
    print(f"  Test Accuracy       : {final_acc:.2f}%")
    print(f"  Conflict Accuracy   : {final_conflict:.2f}%")

    # Save results
    result_dir = os.path.join('./fl_results', args.mode)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, 'results.txt')
    with open(result_path, 'w') as f:
        f.write(f"mode: {args.mode}\n")
        f.write(f"num_clients: {args.num_clients}\n")
        f.write(f"num_rounds: {args.num_rounds}\n")
        f.write(f"local_epochs: {args.local_epochs}\n")
        f.write(f"test_accuracy: {final_acc:.2f}\n")
        f.write(f"conflict_accuracy: {final_conflict:.2f}\n")
    print(f"  Results saved to {result_path}")

    # Save model
    model_path = os.path.join(result_dir, 'global_model_d.pt')
    torch.save(global_model_d.state_dict(), model_path)
    print(f"  Model saved to {model_path}")

    return final_acc, final_conflict


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Federated Learning with Bias Mitigation')

    # FL-specific
    parser.add_argument('--mode', choices=['baseline', 'lff_avg', 'lff_weighted'],
                        required=True, help='Experiment mode')
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--num_rounds', type=int, default=5)
    parser.add_argument('--local_epochs', type=int, default=2)

    # Model / data (reuse existing defaults)
    parser.add_argument('--model', default='MLP', type=str)
    parser.add_argument('--dataset', default='cmnist', type=str)
    parser.add_argument('--percent', default='5pct', type=str)
    parser.add_argument('--data_dir', default='./dataset', type=str)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--q', type=float, default=0.7,
                        help='GCE parameter q')
    parser.add_argument('--ema_alpha', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_attr_idx', type=int, default=0)
    parser.add_argument('--bias_attr_idx', type=int, default=1)

    # Needed by get_backbone but not used for MLP
    parser.add_argument('--train_disent_be', action='store_true')
    parser.add_argument('--resnet_pretrained', action='store_true')

    args = parser.parse_args()

    # Fix seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'#'*60}")
    print(f"  Federated LfF Experiment: {args.mode}")
    print(f"  Clients={args.num_clients}  Rounds={args.num_rounds}  "
          f"LocalEpochs={args.local_epochs}")
    print(f"{'#'*60}\n")

    run_federated(args)
