import copy
import os
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from munch import Munch
import torch.nn.functional as F
from collections import defaultdict
from losses import CCCLoss

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Trainer(object):
    def __init__(self,
                 args,
                 model=None,
                 optimizer=None,
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 fp16_run=False):
        self.args = Munch(args)
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        self.epochs = 0
        self.cccLoss = CCCLoss().to(self.device)

    def save_checkpoint(self, checkpoint_path):
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            "model": self.model}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)


    def _train_epoch(self):
        self.epochs += 1

        train_losses = defaultdict(list)
        self.model.to(self.device)
        self.model.train()

        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]", total=1312), 1):
            batch = [b.to(self.device) for b in batch]
            x_wsp_feat, x_wlm_feat, x_w2v_feat, y_org, y_val, y_arou, y_dom = batch
            sample_size = x_wsp_feat.shape[0]
            random_index = torch.randperm(sample_size).long()
            x_wsp_feat_2 = copy.deepcopy(x_wsp_feat[random_index])
            x_wlm_feat_2 = copy.deepcopy(x_wlm_feat[random_index])
            x_w2v_feat_2 = copy.deepcopy(x_w2v_feat[random_index])
            y_val_2 = copy.deepcopy(y_val[random_index])
            y_arou_2 = copy.deepcopy(y_arou[random_index])

            x_wsp_feat.requires_grad_()
            x_wlm_feat.requires_grad_()
            x_w2v_feat.requires_grad_()

            x_wsp_feat_2.requires_grad_()
            x_wlm_feat_2.requires_grad_()
            x_w2v_feat_2.requires_grad_()

            self.optimizer.zero_grad()

            preds = self.model.forward(x_wsp_feat.float(), x_wlm_feat.float(), x_w2v_feat.float())
            val_loss = F.l1_loss(preds[:,0],y_val)
            arou_loss = F.l1_loss(preds[:,1], y_arou)

            preds_2 = self.model.forward(x_wsp_feat_2.float(), x_wlm_feat_2.float(), x_w2v_feat_2.float())
            val_loss_2 = F.l1_loss(preds_2[:, 0], y_val_2)
            arou_loss_2 = F.l1_loss(preds_2[:, 1], y_arou_2)

            val_rank = [torch.tensor(self.return_rank(elem)) for elem in (y_val-y_val_2)]
            val_rank = torch.stack(val_rank).to(self.device)
            arou_rank = [torch.tensor(self.return_rank(elem)) for elem in (y_arou - y_arou_2)]
            arou_rank = torch.stack(arou_rank).to(self.device)

            val_prob = torch.exp(preds[:, 0] - preds_2[:,0]) / (1 + torch.exp(preds[:, 0] - preds_2[:,0]))
            val_rank_loss = (- val_rank * torch.log(val_prob) - (1 - val_rank)*torch.log(1-val_prob)).mean()

            arou_prob = torch.exp(preds[:, 1] - preds_2[:, 1]) / (1 + torch.exp(preds[:, 1] - preds_2[:, 1]))
            arou_rank_loss = (- arou_rank * torch.log(arou_prob) - (1 - arou_rank) * torch.log(1 - arou_prob)).mean()

            if self.args.use_mixup:
                prob = random.random()
                if prob <= self.args.mixup_prob:
                    lamda = random.random()
                    x_wsp_mix_feat = lamda * x_wsp_feat + (1-lamda) * x_wsp_feat_2
                    x_wlm_mix_feat = lamda * x_wlm_feat + (1-lamda) * x_wlm_feat_2
                    x_w2v_mix_feat = lamda * x_w2v_feat + (1-lamda) * x_w2v_feat_2

                    y_val_mix = lamda * y_val + (1-lamda) * y_val_2
                    y_arou_mix = lamda * y_arou + (1-lamda) * y_arou_2

                    pred_mix = self.model.forward(x_wsp_mix_feat.float(), x_wlm_mix_feat.float(), x_w2v_mix_feat.float())
                    val_mixup_loss = F.l1_loss(pred_mix[:,0],y_val_mix)
                    arou_mixup_loss = F.l1_loss(pred_mix[:,1], y_arou_mix)
                else:
                    val_mixup_loss = torch.zeros(1).mean()
                    arou_mixup_loss = torch.zeros(1).mean()
            else:
                val_mixup_loss = torch.zeros(1).mean()
                arou_mixup_loss = torch.zeros(1).mean()


            loss = self.args.beta * (val_loss + arou_loss + val_loss_2 + arou_loss_2) \
                   +  (1-self.args.beta)*(val_rank_loss + arou_rank_loss) \
                   + self.args.lambda_mixup * (val_mixup_loss + arou_mixup_loss)
            loss.backward()

            self.optimizer.step()
            train_losses["train/val_loss"].append((val_loss + val_loss_2).item())
            train_losses["train/arou_loss"].append((arou_loss + arou_loss_2).item())
            train_losses["train/rank_loss"].append((val_rank_loss + arou_rank_loss).item())
            train_losses["train/mixup_loss"].append((val_mixup_loss + arou_mixup_loss).item())

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        eval_losses = defaultdict(list)

        self.model.eval()
        self.model.to(self.device)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]", total=479), 1):
            ### load data
            batch = [b.to(self.device) for b in batch]
            x_wsp_feat, x_wlm_feat, x_w2v_feat, y_org, y_val, y_arou, y_dom = batch

            preds =  self.model.forward(x_wsp_feat.float(), x_wlm_feat.float(), x_w2v_feat.float())
            val_loss = F.l1_loss(preds[:, 0], y_val)
            val_ccc_loss = self.cccLoss(preds[:, 0], y_val)

            arou_loss = F.smooth_l1_loss(preds[:, 1], y_arou)
            arou_ccc_loss = self.cccLoss(preds[:, 1], y_arou)


            eval_losses["eval/val_loss"].append(val_loss.item())
            eval_losses["eval/val_ccc_loss"].append(val_ccc_loss.item())
            eval_losses["eval/arou_loss"].append(arou_loss.item())
            eval_losses["eval/arou_ccc_loss"].append(arou_ccc_loss.item())

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        return eval_losses


    @torch.no_grad()
    def return_rank(self, y1):
        if y1 < 0:
            return 0.0
        if y1 == 0:
            return 0.5
        else:
            return 1.0

class ClassifierTrainer(object):
    def __init__(self,
                 args,
                 model=None,
                 optimizer=None,
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 fp16_run=False):
        self.args = Munch(args)
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        self.epochs = 0
        self.cccLoss = CCCLoss().to(self.device)

    def save_checkpoint(self, checkpoint_path):
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            "model": self.model}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)


    def _train_epoch(self):
        self.epochs += 1

        train_losses = defaultdict(list)
        self.model.to(self.device)
        self.model.train()

        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]", total=1312), 1):
            batch = [b.to(self.device) for b in batch]
            x_wsp_feat, x_wlm_feat, x_w2v_feat, y_org, y_val, y_arou, y_dom = batch

            x_wsp_feat.requires_grad_()
            x_wlm_feat.requires_grad_()
            x_w2v_feat.requires_grad_()

            self.optimizer.zero_grad()

            preds = self.model.forward(x_wsp_feat.float(), x_wlm_feat.float(), x_w2v_feat.float())
            cls_loss = F.cross_entropy(preds, y_org, weight=torch.tensor(self.args.class_weights).to(preds.device))
            cls_acc = self.accuracy(preds.detach(), y_org)

            loss = cls_loss
            loss.backward()

            self.optimizer.step()
            train_losses["train/clasification_loss"].append(cls_loss.item())
            train_losses["train/clasification_acc"].append(cls_acc.item())

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        eval_losses = defaultdict(list)

        self.model.eval()
        self.model.to(self.device)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]", total=479), 1):
            ### load data
            batch = [b.to(self.device) for b in batch]
            x_wsp_feat, x_wlm_feat, x_w2v_feat, y_org, y_val, y_arou, y_dom = batch

            preds =  self.model.forward(x_wsp_feat.float(), x_wlm_feat.float(), x_w2v_feat.float())
            cls_loss = F.cross_entropy(preds, y_org, weight=torch.tensor(self.args.class_weights).to(preds.device))
            cls_acc = self.accuracy(preds.detach(), y_org)

            eval_losses["eval/clasification_loss"].append(cls_loss.item())
            eval_losses["eval/clasification_acc"].append(cls_acc.item())

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        return eval_losses

    def accuracy(self, predictions, labels):
        classes = torch.argmax(predictions, dim=1)
        return torch.mean((classes == labels).float())


class ClassifierRegressorTrainer(object):
    def __init__(self,
                 args,
                 model=None,
                 optimizer=None,
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 fp16_run=False):
        self.args = Munch(args)
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        self.epochs = 0
        self.cccLoss = CCCLoss().to(self.device)

    def save_checkpoint(self, checkpoint_path):
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            "model": self.model}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)


    def _train_epoch(self):
        self.epochs += 1

        train_losses = defaultdict(list)
        self.model.to(self.device)
        self.model.train()

        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]", total=1312), 1):
            batch = [b.to(self.device) for b in batch]
            x_wsp_feat, x_wlm_feat, x_w2v_feat, y_org, y_val, y_arou, y_dom = batch
            sample_size = x_wsp_feat.shape[0]
            random_index = torch.randperm(sample_size).long()
            x_wsp_feat_2 = copy.deepcopy(x_wsp_feat[random_index])
            x_wlm_feat_2 = copy.deepcopy(x_wlm_feat[random_index])
            x_w2v_feat_2 = copy.deepcopy(x_w2v_feat[random_index])
            y_val_2 = copy.deepcopy(y_val[random_index])
            y_arou_2 = copy.deepcopy(y_arou[random_index])

            x_wsp_feat.requires_grad_()
            x_wlm_feat.requires_grad_()
            x_w2v_feat.requires_grad_()

            x_wsp_feat_2.requires_grad_()
            x_wlm_feat_2.requires_grad_()
            x_w2v_feat_2.requires_grad_()

            self.optimizer.zero_grad()

            preds, cls_pred = self.model.forward(x_wsp_feat.float(), x_wlm_feat.float(), x_w2v_feat.float())
            val_loss = F.l1_loss(preds[:,0],y_val)
            arou_loss = F.l1_loss(preds[:,1], y_arou)

            cls_loss = F.cross_entropy(cls_pred, y_org, weight=torch.tensor(self.args.class_weights).to(cls_pred.device))
            cls_acc = self.accuracy(cls_pred.detach(), y_org)


            preds_2, _ = self.model.forward(x_wsp_feat_2.float(), x_wlm_feat_2.float(), x_w2v_feat_2.float())
            val_loss_2 = F.l1_loss(preds_2[:, 0], y_val_2)
            arou_loss_2 = F.l1_loss(preds_2[:, 1], y_arou_2)

            val_rank = [torch.tensor(self.return_rank(elem)) for elem in (y_val-y_val_2)]
            val_rank = torch.stack(val_rank).to(self.device)
            arou_rank = [torch.tensor(self.return_rank(elem)) for elem in (y_arou - y_arou_2)]
            arou_rank = torch.stack(arou_rank).to(self.device)

            val_prob = torch.exp(preds[:, 0] - preds_2[:,0]) / (1 + torch.exp(preds[:, 0] - preds_2[:,0]))
            val_rank_loss = (- val_rank * torch.log(val_prob) - (1 - val_rank)*torch.log(1-val_prob)).mean()

            arou_prob = torch.exp(preds[:, 1] - preds_2[:, 1]) / (1 + torch.exp(preds[:, 1] - preds_2[:, 1]))
            arou_rank_loss = (- arou_rank * torch.log(arou_prob) - (1 - arou_rank) * torch.log(1 - arou_prob)).mean()

            if self.args.use_mixup:
                prob = random.random()
                if prob <= self.args.mixup_prob:
                    lamda = random.random()
                    x_wsp_mix_feat = lamda * x_wsp_feat + (1-lamda) * x_wsp_feat_2
                    x_wlm_mix_feat = lamda * x_wlm_feat + (1-lamda) * x_wlm_feat_2
                    x_w2v_mix_feat = lamda * x_w2v_feat + (1-lamda) * x_w2v_feat_2

                    y_val_mix = lamda * y_val + (1-lamda) * y_val_2
                    y_arou_mix = lamda * y_arou + (1-lamda) * y_arou_2

                    pred_mix, _ = self.model.forward(x_wsp_mix_feat.float(), x_wlm_mix_feat.float(), x_w2v_mix_feat.float())
                    val_mixup_loss = F.l1_loss(pred_mix[:,0],y_val_mix)
                    arou_mixup_loss = F.l1_loss(pred_mix[:,1], y_arou_mix)
                else:
                    val_mixup_loss = torch.zeros(1).mean()
                    arou_mixup_loss = torch.zeros(1).mean()
            else:
                val_mixup_loss = torch.zeros(1).mean()
                arou_mixup_loss = torch.zeros(1).mean()


            loss = self.args.beta * (val_loss + arou_loss + val_loss_2 + arou_loss_2) \
                   +  (1-self.args.beta)*(val_rank_loss + arou_rank_loss) \
                   + self.args.lambda_mixup * (val_mixup_loss + arou_mixup_loss) \
                   + self.args.lambda_cls * cls_loss
            loss.backward()

            self.optimizer.step()
            train_losses["train/val_loss"].append((val_loss + val_loss_2).item())
            train_losses["train/arou_loss"].append((arou_loss + arou_loss_2).item())
            train_losses["train/rank_loss"].append((val_rank_loss + arou_rank_loss).item())
            train_losses["train/mixup_loss"].append((val_mixup_loss + arou_mixup_loss).item())
            train_losses["train/clasification_loss"].append(cls_loss.item())
            train_losses["train/clasification_acc"].append(cls_acc.item())

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        eval_losses = defaultdict(list)

        self.model.eval()
        self.model.to(self.device)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]", total=479), 1):
            ### load data
            batch = [b.to(self.device) for b in batch]
            x_wsp_feat, x_wlm_feat, x_w2v_feat, y_org, y_val, y_arou, y_dom = batch

            preds, cls_pred =  self.model.forward(x_wsp_feat.float(), x_wlm_feat.float(), x_w2v_feat.float())
            val_loss = F.l1_loss(preds[:, 0], y_val)
            val_ccc_loss = self.cccLoss(preds[:, 0], y_val)

            arou_loss = F.smooth_l1_loss(preds[:, 1], y_arou)
            arou_ccc_loss = self.cccLoss(preds[:, 1], y_arou)

            cls_loss = F.cross_entropy(cls_pred, y_org, weight=torch.tensor(self.args.class_weights).to(cls_pred.device))
            cls_acc = self.accuracy(cls_pred.detach(), y_org)

            eval_losses["eval/val_loss"].append(val_loss.item())
            eval_losses["eval/val_ccc_loss"].append(val_ccc_loss.item())
            eval_losses["eval/arou_loss"].append(arou_loss.item())
            eval_losses["eval/arou_ccc_loss"].append(arou_ccc_loss.item())
            eval_losses["eval/clasification_loss"].append(cls_loss.item())
            eval_losses["eval/clasification_acc"].append(cls_acc.item())

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        return eval_losses


    @torch.no_grad()
    def return_rank(self, y1):
        if y1 < 0:
            return 0.0
        if y1 == 0:
            return 0.5
        else:
            return 1.0

    def accuracy(self, predictions, labels):
        classes = torch.argmax(predictions, dim=1)
        return torch.mean((classes == labels).float())