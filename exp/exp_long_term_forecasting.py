from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                if is_test:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                # loss = criterion(outputs, batch_y)

                # loss = loss.detach().cpu()
                tgt = self.args.target_var_idx
                #pred = outputs[:, :, tgt:tgt+1].detach().cpu()
                pred = outputs[:, :, -1:].detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()   
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                def _assert_finite(name, t):
                    if not torch.isfinite(t).all():
                        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
                        raise RuntimeError(f"{name} contains non-finite values at indices (showing first 5): {bad}")

                # In train() minibatch loop, after moving to device:
                _assert_finite("batch_x", batch_x)
                _assert_finite("batch_y", batch_y)
                _assert_finite("batch_x_mark", batch_x_mark)
                _assert_finite("batch_y_mark", batch_y_mark)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                        loss = criterion(outputs, batch_y)                        
                        loss_val += loss.item()
                        count += 1
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    # print("The output for this batch is: ",outputs)
                    if not torch.isfinite(outputs).all():
                        raise RuntimeError("Model outputs contain NaN/Inf")
                    # loss = criterion(outputs, batch_y)
                    tgt = self.args.target_var_idx
                    #outputs_y = outputs[:, :, tgt:tgt+1]
                    outputs_y = outputs[:, :, -1:]   
                    # print("outputs_y is: ", outputs_y)
                    # print("batch_y is: ", batch_y)
                    loss = criterion(outputs_y, batch_y)
                    # print("Loss for this batch is: ", loss.item())
                    loss_val += loss.item()
                    count += 1
                
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))   
            if self.args.use_multi_gpu:
                dist.barrier()   
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)      
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # how many autoregressive steps do we need
                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1

                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        # shift encoder window and marks by one token, append last predicted token
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j-1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    # take the last token_len steps from the model output
                    pred_y.append(outputs[:, -self.args.token_len:, :])

                # concat all predicted tokens along time
                pred_y = torch.cat(pred_y, dim=1)  # [B, steps*token_len, C]
                if dis != 0:
                    # trim extra steps so length == test_pred_len
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]

                # align ground-truth target window to the same horizon
                batch_y = batch_y[:, -self.args.test_pred_len:, :]  # [B, T_pred, 1]

                # --- slice the target channel from the autoregressive predictions ---
                tgt = self.args.target_var_idx  # e.g., -1 means last channel
                if tgt < 0:
                    tgt = pred_y.shape[-1] + tgt
                outputs_y = pred_y[:, :, tgt:tgt+1]  # [B, T_pred, 1]

                # safety: time dimension must match
                assert outputs_y.shape[1] == batch_y.shape[1], f"time mismatch: pred {outputs_y.shape} vs true {batch_y.shape}"

                # detach for accumulation
                pred = outputs_y.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)

                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                if self.args.visualize and i == 0:
                    gt = np.array(true[0, :, 0])
                    pd = np.array(pred[0, :, 0])
                    lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    visual(gt, pd, os.path.join(dir_path, f'{i}.png'))

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        np.save(os.path.join(folder_path, 'predictions.npy'), preds)
        np.save(os.path.join(folder_path, 'ground_truth.npy'), trues)

        mae, mse, rmse, mape, mspe, r2, kelly_r2 = metric(preds, trues)
        print(f"MSE: {mse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}  Kelly R²: {kelly_r2:.4f}")
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, r2:{}, kelly_r2:{}'.format(mse, mae, r2, kelly_r2))
            f.write('\n\n')
        return
# comment for git diff