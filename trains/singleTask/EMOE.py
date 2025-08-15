import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str, eva_imp, uni_distill, entropy_balance

logger = logging.getLogger('EMOE')

class EMOE():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        params = list(model.parameters())
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0

            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision)
                    w = output['channel_weight']

                    y_pred.append(output['logits_c'].cpu())
                    y_true.append(labels.cpu())

                    loss_task_l = self.criterion(output['logits_l'], labels)
                    loss_task_v = self.criterion(output['logits_v'], labels)
                    loss_task_a = self.criterion(output['logits_a'], labels)
                    loss_task_m = self.criterion(output['logits_c'], labels)

                    l_dist = eva_imp(output['logits_l'], labels)
                    a_dist = eva_imp(output['logits_a'], labels)
                    v_dist = eva_imp(output['logits_v'], labels)
                    dist = torch.zeros(l_dist.shape[0], 3).to(self.args.device)
                    for i,_ in enumerate(l_dist):
                        s = 1/(l_dist[i]+0.1) + 1/(v_dist[i]+0.1) + 1/(a_dist[i]+0.1)
                        dist[i][0] = (1/(l_dist[i]+0.1)) / s
                        dist[i][1] = (1/(v_dist[i]+0.1)) / s
                        dist[i][2] = (1/(a_dist[i]+0.1)) / s
                    loss_sim = torch.mean(torch.mean((dist.detach() - w) ** 2, dim=-1))
                    loss_ety = entropy_balance(w)

                    if self.args.fusion_method == "sum":
                        loss_ud = uni_distill(output['c_proj'], (output['l_proj'] * w[:,0].view(-1, 1) + output['v_proj'] * w[:,1].view(-1, 1) + 
                        output['a_proj'] * w[:,2].view(-1, 1)).detach())
                    elif self.args.fusion_method == "concat":
                        loss_ud = uni_distill(output['c_proj'], torch.cat([output['l_proj'] * w[:,0].view(-1, 1),output['v_proj'] * w[:,1].view(-1, 1),
                        output['a_proj'] * w[:,2].view(-1, 1)], dim=1).detach())

                    loss = loss_task_m + (loss_task_l + loss_task_v + loss_task_a)/3 + 0.1*(loss_ety + 0.1*loss_sim) + 0.1*loss_ud

                    loss.backward()
                    train_loss += loss.item()

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            torch.save(model.state_dict(), './pt/' + str(epochs) + '.pth')
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = './pt/emoe.pth'
                torch.save(model.state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False, f=0):
        model.eval()
        y_pred, y_true = [], []
        weight, ability = [], []
        c_fea = []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision)

                    loss = self.criterion(output['logits_c'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['logits_c'].cpu())
                    y_true.append(labels.cpu())
                    
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        
        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results