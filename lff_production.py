from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from data.util import get_dataset, IdxDataset
from module.loss import GeneralizedCELoss
from module.util import get_model
from module.unet import *
from module.util import get_backbone
from util import *
import warnings
warnings.filterwarnings(action='ignore')
import copy
class Learner(object):
    def __init__(self, args):
        self.args = args

        data2model = {'cmnist': args.model,
                      'bar': "ResNet18",
                      'bffhq': "ResNet18",
                      'dogs_and_cats': "ResNet18",
                      'cifar10c':"ResNet18",
                      'wbirds':"ResNet18",
                     }

        data2batch_size = {'cmnist': 256,
                           'bar': 64,
                           'bffhq': 64,
                           'wbirds':64,
                           'dogs_and_cats': 64,
                           'cifar10c':256,
                          }
        
        data2preprocess = {'cmnist': None,
                           'bar': True,
                           'bffhq': True,
                           'dogs_and_cats':True,
                           'wbirds':True,
                           'cifar10c':True,

                          }

        run_name = args.exp
        
        self.model = data2model[args.dataset]
        self.batch_size = data2batch_size[args.dataset]

        print(f'model: {self.model} || dataset: {args.dataset}')
        print(f'working with experiment: {args.exp}...')
        self.log_dir = os.makedirs(os.path.join(args.log_dir, args.dataset, args.exp), exist_ok=True)
        self.device = torch.device(args.device)
        self.args = args

        print(self.args)

        # logging directories
        self.log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
        self.summary_dir =  os.path.join(args.log_dir, args.dataset, args.tensorboard_dir, args.exp)
        self.result_dir = os.path.join(self.log_dir, "result")
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
            
        self.train_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="train",
            transform_split="train",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
        )
        self.valid_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="valid",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
        )

        self.test_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="test",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
        )

        train_target_attr = []
        for data in self.train_dataset.data:
            train_target_attr.append(int(data.split('_')[-2]))
        train_target_attr = torch.LongTensor(train_target_attr)

        attr_dims = []
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        self.num_classes = attr_dims[0]

        self.train_dataset = IdxDataset(self.train_dataset)

        # make loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # define model and optimizer
        self.model_b = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
        self.model_d = get_backbone(self.model, self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)

        self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        self.optimizer_d = torch.optim.Adam(
                self.model_d.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        # define loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        print(f'self.criterion: {self.criterion}')

        self.bias_criterion = GeneralizedCELoss(q=0.7)
        print(f'self.bias_criterion: {self.bias_criterion}')

        self.sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha)
        self.sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha)

        print(f'alpha : {self.sample_loss_ema_d.alpha}')
        self.best_valid_acc_b, self.best_test_acc_b = 0., 0.
        self.best_valid_acc_d, self.best_test_acc_d = 0., 0.

        print('finished model initialization....')

    # evaluation code for vanilla
    def evaluate(self, model, data_loader):
        model.eval()
        total_correct, total_num = 0, 0
        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct/float(total_num)
        model.train()
        return accs

    def evaluate_dual_models(self, data_loader):
        self.model_b.eval()
        self.model_d.eval()

        metrics = {
            "total": 0, "align": 0, "conflict": 0,
            "loss_d": 0.0, "loss_b": 0.0,
            "acc_d_total": 0, "acc_b_total": 0,
            "acc_d_align": 0, "acc_d_conflict": 0,
            "acc_b_align": 0, "acc_b_conflict": 0,
        }
        with torch.no_grad():

            for data, attr, path in data_loader:
                data = data.to(self.device)
                target_label = attr[:, self.args.target_attr_idx].to(self.device)
                bias_label = attr[:, self.args.bias_attr_idx].to(self.device)
                logit_d = self.model_d(data)
                loss_d = self.criterion(logit_d, target_label).mean()
                pred_d = logit_d.data.max(1, keepdim=True)[1].squeeze(1)

                # Model B
                logit_b = self.model_b(data)
                loss_b = self.bias_criterion(logit_b, target_label).mean()
                pred_b = logit_b.data.max(1, keepdim=True)[1].squeeze(1)
                # Metrics
                is_align = (target_label == bias_label)
                is_conflict = (target_label != bias_label)

                correct_d = (pred_d == target_label)
                correct_b = (pred_b == target_label)

                metrics["total"] += data.size(0)
                metrics["loss_d"] += loss_d.item() * data.size(0)
                metrics["loss_b"] += loss_b.item() * data.size(0)

                metrics["acc_d_total"] += correct_d.sum().item()
                metrics["acc_b_total"] += correct_b.sum().item()

                metrics["align"] += is_align.sum().item()
                metrics["conflict"] += is_conflict.sum().item()

                metrics["acc_d_align"] += (correct_d & is_align).sum().item()
                metrics["acc_d_conflict"] += (correct_d & is_conflict).sum().item()

                metrics["acc_b_align"] += (correct_b & is_align).sum().item()
                metrics["acc_b_conflict"] += (correct_b & is_conflict).sum().item()

                #Formatting

        total = max (metrics["total"], 1)
        t_align = max (metrics["align"], 1)
        t_conflict = max (metrics["conflict"], 1)

        self.model_b.train()
        self.model_d.train()


        return {
            "loss_d": metrics["loss_d"]/total,
            "loss_b": metrics["loss_b"]/total,
            "acc_d_total": (metrics["acc_d_total"]/total)*100,
            "acc_b_total": (metrics["acc_b_total"]/total)*100,
            "acc_d_align": (metrics["acc_d_align"]/t_align)*100,
            "acc_d_conflict": (metrics["acc_d_conflict"]/t_conflict)*100,
            "acc_b_align": (metrics["acc_b_align"]/t_align)*100,
            "acc_b_conflict": (metrics["acc_b_conflict"]/t_conflict)*100,
        }

    def log_dual_results(self, path, epoch, split, res):
        with open(path, "a") as f:
            f.write(f"{epoch}, {split}, {res['loss_d']:.4f}, {res['loss_b']:.4f},"
                    f"{res['acc_d_total']:.2f}, {res['acc_b_total']:.2f},"
                    f"{res['acc_d_conflict']:.2f}, {res['acc_b_conflict']:.2f},"
                    f"{res['acc_d_align']:.2f}, {res['acc_b_align']:.2f}\n")



        
        

    


    def save_best_d(self, epoch):
        model_path = os.path.join(self.result_dir, f"best_model_d_epoch{epoch}.th")
        torch.save({
            'epoch': epoch,
            'state_dict': self.model_d.state_dict(),
            'optimizer': self.optimizer_d.state_dict(),
        }, model_path)
        print(f"[SAVE] best model_d saved -> {model_path}")


    def save_best_b(self, epoch):
        model_path = os.path.join(self.result_dir, f"best_model_b_epoch{epoch}.th")
        torch.save({
            'epoch': epoch,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }, model_path)
        print(f"[SAVE] best model_b saved -> {model_path}")

    def evaluate_log_save(self, epoch, log_path):
        # evaluate
        val_res  = self.evaluate_dual_models(self.valid_loader)
        test_res = self.evaluate_dual_models(self.test_loader)

        # log
        # self.log_dual_results(log_path, epoch, "valid", val_res)
        self.log_dual_results(log_path, epoch, "test", test_res)

        # save model_b based on best acc_b_total (test)
        if test_res["acc_b_total"] >= self.best_test_acc_b:
            self.best_test_acc_b = test_res["acc_b_total"]
            self.save_best_b(epoch)

        # save model_d based on best acc_d_total (test)
        if test_res["acc_d_total"] >= self.best_test_acc_d:
            self.best_test_acc_d = test_res["acc_d_total"]
            self.save_best_d(epoch)

        # print summary
        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"[VALID] acc_d_total={val_res['acc_d_total']:.2f} | acc_b_total={val_res['acc_b_total']:.2f}")
        print(f"[TEST ] acc_d_total={test_res['acc_d_total']:.2f} | acc_b_total={test_res['acc_b_total']:.2f}")
        print(f"       acc_d_conflict={test_res['acc_d_conflict']:.2f} | acc_b_conflict={test_res['acc_b_conflict']:.2f}")

        return val_res, test_res


    def save_best(self, step):
        model_path = os.path.join(self.result_dir, f"best_model_d_{step}.th")
        state_dict = {
            'steps': step,
            'state_dict': self.model_d.state_dict(),
            'optimizer': self.optimizer_d.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        model_path = os.path.join(self.result_dir, f"best_model_b_{step}.th")
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{step} model saved ...')


    def board_lff_acc(self, step, inference=None):
        # check label network
        valid_accs_b = self.evaluate(self.model_b, self.valid_loader)
        test_accs_b = self.evaluate(self.model_b, self.test_loader)

        valid_accs_d = self.evaluate(self.model_d, self.valid_loader)
        test_accs_d = self.evaluate(self.model_d, self.test_loader)

        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        if valid_accs_b >= self.best_valid_acc_b:
            self.best_valid_acc_b = valid_accs_b

        if test_accs_b >= self.best_test_acc_b:
            self.best_test_acc_b = test_accs_b

        if valid_accs_d >= self.best_valid_acc_d:
            self.best_valid_acc_d = valid_accs_d

        if test_accs_d >= self.best_test_acc_d:
            self.best_test_acc_d = test_accs_d
            self.save_best(step)

        
        print(f'valid_b: {valid_accs_b} || test_b: {test_accs_b} ')
        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')


    def concat_dummy(self, z):
        def hook(model, input, output):
            z.append(output.squeeze())
            return torch.cat((output, torch.zeros_like(output)), dim=1)
        return hook

   

    def train_lff_unaware(self, args):
        print('\n' + '='*60)
        print('STRATEGY: UNAWARE LfF')
        print('='*60)

        num_updated = 0
        train_iter = iter(self.train_loader)
        train_num = len(self.train_dataset.dataset)
        epoch, cnt = 0, 0

        log_path = os.path.join(self.result_dir, "training_log.csv")
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("epoch, split, loss_d, loss_b, acc_d_total, acc_b_total, acc_d_conflict, acc_b_conflict, acc_d_align, acc_b_align\n")
 
        metrics_history = { 'loss_b': [], 'loss_d': [],'weight_conflict': [], 'weight_align': [] }



        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=args.lr_decay_step,gamma=args.lr_gamma)

        for step in tqdm(range(args.num_steps)):
            # train main model
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            index = index.to(self.device)
            label = attr[:, args.target_attr_idx]
            bias_label = attr[:, args.bias_attr_idx]

            logit_b = self.model_b(data)
            logit_d = self.model_d(data)

            loss_b = self.criterion(logit_b, label).cpu().detach()
            loss_d = self.criterion(logit_d, label).cpu().detach()
            loss_b = loss_b.to(self.device)
            loss_d = loss_d.to(self.device)
            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d')

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, index)
            self.sample_loss_ema_d.update(loss_d, index)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

            if np.isnan(loss_b.mean().item()):
                raise NameError('loss_b_ema')
            if np.isnan(loss_d.mean().item()):
                raise NameError('loss_d_ema')

            label_cpu = label.cpu()

            for c in range(self.num_classes):
                class_index = np.where(label_cpu == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c) + 1e-8
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d

            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)

            # logging weights of align and conflicting after every 50th batch

            if step % 50 == 0:
                conf_mask = (bias_label != label)
                if conf_mask.sum() > 0:
                    metrics_history['weight_conflict'].append(loss_weight[conf_mask].mean().item())
                if (~conf_mask).sum() > 0:
                    metrics_history['weight_align'].append(loss_weight[~conf_mask].mean().item())


            loss_b_update = self.bias_criterion(logit_b, label)
            loss_d_update = self.criterion(logit_d, label) * loss_weight.to(self.device)

            if np.isnan(loss_b_update.mean().item()):
                raise NameError('loss_b_update')

            if np.isnan(loss_d_update.mean().item()):
                raise NameError('loss_d_update')

            loss = loss_b_update.mean() + loss_d_update.mean()
            num_updated += loss_weight.mean().item() * data.size(0)

            self.optimizer_b.zero_grad()
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_b.step()
            self.optimizer_d.step()

            if args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_d lr: {self.optimizer_d.param_groups[-1]['lr']}")

            
            cnt += len(index)
            if cnt >= train_num:
                print(f"\n--- Epoch {epoch} Summary ---")

                val_res, test_res = self.evaluate_log_save(epoch, log_path)
                # self.board_lff_acc(step)
                w_conf = np.mean(metrics_history['weight_conflict']) if len(metrics_history['weight_conflict']) > 0 else 0.0
                w_align = np.mean(metrics_history['weight_align']) if len(metrics_history['weight_align']) > 0 else 0.0
                print(f"  [Train] Weights -> Conflict: {w_conf:.4f} | Aligned: {w_align:.4f}")
                print(f"  [Test D] Total Acc: {test_res['acc_d_total']:.2f}% (Conflict: {test_res['acc_d_conflict']:.2f}%)")
                print(f"  [Test B] Total Acc: {test_res['acc_b_total']:.2f}% (Conflict: {test_res['acc_b_conflict']:.2f}%)")


                print(f'finished epoch: {epoch}')
                epoch +=1
                cnt = 0
                metrics_history = { 'loss_b': [], 'loss_d': [],'weight_conflict': [], 'weight_align': [] }


    
    def test_lff_be(self, args):
        if args.dataset == 'cmnist':
            self.model_b = get_backbone("MLP", self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
            self.model_d = get_backbone("MLP", self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
        else:
            self.model_b = get_backbone("ResNet18", self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)
            self.model_d = get_backbone("ResNet18", self.num_classes, args=self.args, pretrained=self.args.resnet_pretrained).to(self.device)

        self.model_d.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_d.th'))['state_dict'])
        self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_b.th'))['state_dict'])
        self.board_lff_acc(step=0, inference=True)

    