#!/usr/bin/env bash
# ============================================================
# Run all 3 Federated Learning experiments sequentially
# ============================================================

set -e  # exit on first error

COMMON_ARGS="--dataset cmnist --percent 5pct --data_dir ./dataset --model MLP \
  --num_clients 5 --num_rounds 5 --local_epochs 2 \
  --lr 0.01 --batch_size 256 --seed 42 --device cuda"

echo "========================================="
echo " Experiment 1: Baseline Federated Model"
echo "========================================="
python train_federated.py --mode baseline $COMMON_ARGS 2>&1 | tee fl_results/baseline.log

echo ""
echo "========================================="
echo " Experiment 2: LfF + Simple FedAvg"
echo "========================================="
python train_federated.py --mode lff_avg $COMMON_ARGS 2>&1 | tee fl_results/lff_avg.log

echo ""
echo "========================================="
echo " Experiment 3: LfF + Accuracy-Weighted"
echo "========================================="
python train_federated.py --mode lff_weighted $COMMON_ARGS 2>&1 | tee fl_results/lff_weighted.log

echo ""
echo "========================================="
echo " All experiments complete!"
echo " Results in: ./fl_results/"
echo "========================================="
