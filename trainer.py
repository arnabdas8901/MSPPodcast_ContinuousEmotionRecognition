import os
import torch
import logging
import random
import numpy as np
from tqdm import tqdm
from munch import Munch
import torch.nn.functional as F
from collections import defaultdict
#from torchmetrics.regression import ConcordanceCorrCoef as CCCLoss
from losses import CCCLoss

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Trainer(object):
    def __init__(self,
                 args,
                 model=None,
                 spk_encoder=None,
                 optimizer=None,
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 fp16_run=False):
        self.args = Munch(args)
        self.model = model
        self.spk_encoder = spk_encoder
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        self.epochs = 0
        self.cccLoss = CCCLoss().to(self.device)
        self.spk_encoder.to(self.device)

    def save_checkpoint(self, checkpoint_path):
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            "model": self.model.state_dict()}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)


    def _train_epoch(self):
        self.epochs += 1

        train_losses = defaultdict(list)

        self.model.train()
        self.model.to(self.device)
        self.spk_encoder.to(self.device)

        scaler = torch.cuda.amp.GradScaler() if (('cuda' in str(self.device)) and self.fp16_run) else None

        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            batch = [b.to(self.device) for b in batch]
            x_real, y_org, y_val, y_arou, y_dom, gender = batch
            x_real.requires_grad_()
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # w2v2 model output arousal, dominance, valence
                    preds= self.model.forward(x_real.float(), gender, self.spk_encoder.encode_batch(x_real).squeeze(1))
                    val_loss = torch.zeros(1).mean()  #F.smooth_l1_loss(preds[:,0],y_val)
                    val_ccc_loss = self.cccLoss(preds[:,-1], y_val)

                    arou_loss = torch.zeros(1).mean()  #F.smooth_l1_loss(preds[:, 1], y_arou)
                    arou_ccc_loss = self.cccLoss(preds[:, 0], y_arou)

                    dom_loss = torch.zeros(1).mean() #F.smooth_l1_loss(preds[:, 2], y_dom)
                    dom_ccc_loss = self.cccLoss(preds[:, 1], y_dom)

                    loss = self.args.lambda_l1 * (val_loss + arou_loss + dom_loss)  \
                           + self.args.lambda_ccc * (val_ccc_loss + arou_ccc_loss + dom_ccc_loss)

                scaler.scale(loss).backward()
            else:
                self.logger.disabled = True
                preds = self.model.forward(x_real.float(), gender, self.spk_encoder.encode_batch(x_real).squeeze(1))
                self.logger.disabled = False
                val_loss = torch.zeros(1).mean()  # F.smooth_l1_loss(preds[:,0],y_val)
                val_ccc_loss = self.cccLoss(preds[:, -1], y_val)

                arou_loss = torch.zeros(1).mean()  # F.smooth_l1_loss(preds[:, 1], y_arou)
                arou_ccc_loss = self.cccLoss(preds[:, 0], y_arou)

                dom_loss = torch.zeros(1).mean()  # F.smooth_l1_loss(preds[:, 2], y_dom)
                dom_ccc_loss = self.cccLoss(preds[:, 1], y_dom)

                loss = self.args.lambda_l1 * (val_loss + arou_loss + dom_loss) \
                       + self.args.lambda_ccc * (val_ccc_loss + arou_ccc_loss + dom_ccc_loss)
                loss.backward()

            if scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            train_losses["train/val_loss"].append(val_loss.item())
            train_losses["train/val_ccc_loss"].append(val_ccc_loss.item())
            train_losses["train/arou_loss"].append(arou_loss.item())
            train_losses["train/arou_ccc_loss"].append(arou_ccc_loss.item())
            train_losses["train/dom_loss"].append(dom_loss.item())
            train_losses["train/dom_ccc_loss"].append(dom_ccc_loss.item())
            train_losses["train/total_loss"].append(loss.item())


        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        eval_losses = defaultdict(list)

        self.model.eval()
        self.model.to(self.device)
        self.spk_encoder.to(self.device)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            ### load data
            batch = [b.to(self.device) for b in batch]
            x_real, y_org, y_val, y_arou, y_dom, gender = batch

            preds =  self.model.forward(x_real.float(), gender, self.spk_encoder.encode_batch(x_real).squeeze(1))
            val_loss = F.smooth_l1_loss(preds[:, -1], y_val)
            val_ccc_loss = self.cccLoss(preds[:, -1], y_val)

            arou_loss = F.smooth_l1_loss(preds[:, 0], y_arou)
            arou_ccc_loss = self.cccLoss(preds[:, 0], y_arou)

            dom_loss = F.smooth_l1_loss(preds[:, 1], y_dom)
            dom_ccc_loss = self.cccLoss(preds[:, 1], y_dom)

            loss = self.args.lambda_l1 * (val_loss + arou_loss + dom_loss) \
                   + self.args.lambda_ccc * (val_ccc_loss + arou_ccc_loss + dom_ccc_loss)

            eval_losses["eval/val_loss"].append(val_loss.item())
            eval_losses["eval/val_ccc_loss"].append(val_ccc_loss.item())
            eval_losses["eval/arou_loss"].append(arou_loss.item())
            eval_losses["eval/arou_ccc_loss"].append(arou_ccc_loss.item())
            eval_losses["eval/dom_loss"].append(dom_loss.item())
            eval_losses["eval/dom_ccc_loss"].append(dom_ccc_loss.item())
            eval_losses["eval/total_loss"].append(loss.item())

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        return eval_losses


class TrainerBaseline(object):
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
            "model": self.model.state_dict()}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)


    def _train_epoch(self):
        self.epochs += 1

        train_losses = defaultdict(list)

        self.model.train()
        self.model.to(self.device)

        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            batch = [b.to(self.device) for b in batch]
            x_real, y_org, y_val, y_arou, y_dom, _ = batch
            #x_real.requires_grad_()
            self.optimizer.zero_grad()
            self.logger.disabled = True
            _, preds = self.model.forward(x_real.float())
            self.logger.disabled = False
            val_loss = F.smooth_l1_loss(preds[:, -1], y_val)
            val_ccc_loss = self.cccLoss(preds[:, -1], y_val)

            arou_loss = F.smooth_l1_loss(preds[:, 0], y_arou)
            arou_ccc_loss = self.cccLoss(preds[:, 0], y_arou)

            dom_loss = F.smooth_l1_loss(preds[:, 1], y_dom)
            dom_ccc_loss = self.cccLoss(preds[:, 1], y_dom)

            loss = self.args.lambda_l1 * (val_loss + arou_loss + dom_loss) \
                   + self.args.lambda_ccc * (val_ccc_loss + arou_ccc_loss + dom_ccc_loss)
            loss.backward()
            self.optimizer.step()
            train_losses["train/val_loss"].append(val_loss.item())
            train_losses["train/val_ccc_loss"].append(val_ccc_loss.item())
            train_losses["train/arou_loss"].append(arou_loss.item())
            train_losses["train/arou_ccc_loss"].append(arou_ccc_loss.item())
            train_losses["train/dom_loss"].append(dom_loss.item())
            train_losses["train/dom_ccc_loss"].append(dom_ccc_loss.item())
            train_losses["train/total_loss"].append(loss.item())


        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        eval_losses = defaultdict(list)

        self.model.eval()
        self.model.to(self.device)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            ### load data
            batch = [b.to(self.device) for b in batch]
            x_real, y_org, y_val, y_arou, y_dom, _ = batch

            _, preds =  self.model.forward(x_real.float())
            val_loss = F.smooth_l1_loss(preds[:, -1], y_val)
            val_ccc_loss = self.cccLoss(preds[:, -1], y_val)

            arou_loss = F.smooth_l1_loss(preds[:, 0], y_arou)
            arou_ccc_loss = self.cccLoss(preds[:, 0], y_arou)


            eval_losses["eval/val_loss"].append(val_loss.item())
            eval_losses["eval/val_ccc_loss"].append(val_ccc_loss.item())
            eval_losses["eval/arou_loss"].append(arou_loss.item())
            eval_losses["eval/arou_ccc_loss"].append(arou_ccc_loss.item())
            eval_losses["eval/total_ccc_loss"].append((val_ccc_loss+ arou_ccc_loss).item())

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        return eval_losses