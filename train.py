import numpy as np
import torch
import random
from lff_production import Learner as LFF_Learner
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training LFF models')

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the CUDA version used by PyTorch
        print(f"CUDA version used by PyTorch: {torch.version.cuda}")
        # Get the current device's CUDA capability
        print(f"CUDA capability of the current device: {torch.cuda.get_device_capability()}")
        # Print the device name
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")

    # training
    parser.add_argument("--batch_size", help="batch_size", default=256, type=int)
    parser.add_argument("--lr",help='learning rate',default=1e-3, type=float)
    parser.add_argument("--weight_decay",help='weight_decay',default=0.0, type=float)
    parser.add_argument("--momentum",help='momentum',default=0.9, type=float)
    parser.add_argument("--num_workers", help="workers number", default=16, type=int)
    parser.add_argument("--exp", help='experiment name', default='debugging', type=str)
    parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)
    parser.add_argument("--num_steps", help="# of iterations", default= (500 * 100), type=int)
    parser.add_argument("--target_attr_idx", help="target_attr_idx", default= 0, type=int)
    parser.add_argument("--bias_attr_idx", help="bias_attr_idx", default= 1, type=int)
    parser.add_argument("--dataset", help="data to train", default= 'cmnist', type=str)
    parser.add_argument("--percent", help="percentage of conflict", default= "1pct", type=str)
    parser.add_argument("--use_lr_decay", action='store_true', help="whether to use learning rate decay")
    parser.add_argument("--lr_decay_step", help="learning rate decay steps", type=int, default=10000)
    parser.add_argument("--q", help="GCE parameter q", type=float, default=0.7)
    parser.add_argument("--lr_gamma",  help="lr gamma", type=float, default=0.1)
    parser.add_argument("--lambda_dis_align",  help="lambda_dis in Eq.2 of DisEnt", type=float, default=1.0)
    parser.add_argument("--lambda_swap_align",  help="lambda_swap_b in Eq.3 of DisEnt", type=float, default=1.0)
    parser.add_argument("--lambda_swap",  help="lambda swap (lambda_swap in Eq.4 of DisEnt)", type=float, default=1.0)
    parser.add_argument("--ema_alpha",  help="use weight mul", type=float, default=0.7)
    parser.add_argument("--curr_step", help="curriculum steps", type=int, default= 0)
    parser.add_argument("--model", help="which network, [MLP, ResNet18]", default= 'MLP', type=str)
    parser.add_argument("--tensorboard_dir", help="tensorboard directory", default= 'summary', type=str)
    parser.add_argument("--lr_decay", action="store_true") 

    # logging
    parser.add_argument("--log_dir", help='path for saving model', default='./log', type=str)
    parser.add_argument("--data_dir", help='path for loading data', default='./dataset', type=str)
    parser.add_argument("--valid_freq", help='frequency to evaluate on valid/test set', default=500, type=int)
    parser.add_argument("--log_freq", help='frequency to log on tensorboard', default=500, type=int)
    parser.add_argument("--tensorboard", action="store_true", help="whether to use tensorboard")

    # experiment
    parser.add_argument("--train_lff_be", action="store_true", help="whether to train lff with our selection method")
    parser.add_argument("--train_disent_be", action="store_true", help="whether to train disent with our selection method")
    parser.add_argument("--train_bias_amplification_unaware", action="store_true", help="whether to train bias amplification unaware model")
    parser.add_argument("--train_lff_unaware", action="store_true", help="whether to train lff unaware model")
    parser.add_argument("--fix_randomseed", action="store_true", help="fix randomseed")
    parser.add_argument("--seed",  help="seed", type=int, default=42)
    parser.add_argument("--biased_model_train_iter", type=int, default=1000, help="biased_model_stop iteration")
    parser.add_argument("--biased_model_softmax_threshold", type=float, default=0.99, help="biased_model_softmax_threshold")
    parser.add_argument("--num_bias_models", type=int, default=5, help="number of bias models")
    parser.add_argument("--resnet_pretrained", action="store_true", help="use pretrained ResNet")
    parser.add_argument("--agreement", type=int, default=3, help="number of agreement")
    
    parser.add_argument("--clamp_value", help = 'torch clamp the loss between range', default=0.7, type = float)
    parser.add_argument("--pretrained_path", help='path for pretrained model', type=str)
    parser.add_argument("--thresh", default=0.97, help="Threshold for the MixUP") # number of bias conflicting samples it is going to take.  essentailly we remove that much percentage of sampels from potential conflicting samples list.
    parser.add_argument("--mix_up_val", default = 0.9, type = float, help = "Mix up value") # alpha parameter accoridng to paper. 
    parser.add_argument("--loss_contr", default=0.5, help="Loss contribution", type = float) # loss contribution of hybrid samples. 

    args = parser.parse_args()
    if args.fix_randomseed:
        random_seed = args.seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    # init learner
    
        learner_baseline = LFF_Learner(args)

    # actual training
    print('Training starts ...')


    if args.train_lff_unaware:
        learner_baseline.train_lff_unaware(args)
    else:
        print('choose one of the two options ...')
        import sys
        sys.exit(0)
