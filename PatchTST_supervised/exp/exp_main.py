from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from models_diffusion import DDPM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric, GLUONTS_METRICS

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import wandb
import copy
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'TimeDiff': DDPM,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def pretrain(self, setting):

        path = os.path.join(self.args.checkpoints, f'{setting}_{self.args.time}')
        if not os.path.exists(path):
            os.makedirs(path)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = optim.Adam(self.model.parameters(), lr=0.0001)

        best_train_loss = 10000000.0
        for epoch in range(self.args.pretrain_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = batch_y
                loss = self.model.pretrain_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            print("PreTraining Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("PreTraining Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ".format(epoch + 1, train_steps, train_loss))

            if train_loss < best_train_loss:
                print("-------------------------")
                best_train_loss = train_loss
                torch.save(self.model.dlinear_model.state_dict(), path + '/' + 'pretrain_model_checkpoint.pth')



    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # print(f'vali batch {i}, batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}')
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    if self.args.model == 'TimeDiff':
                        outputs,_,_,_,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                    
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, f'{setting}_{self.args.time}')
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        total_steps = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # print(f'train epoch {epoch}, batch {i},batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}，batch_x_mark shape: {batch_x_mark.shape}, batch_y_mark shape: {batch_y_mark.shape}')
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif self.args.model == 'TimeDiff':
                        print('TimeDiff predict...')
                        outputs,_,_,_,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    # print(f'outputs shape: {outputs.shape}, batch_y shape: {batch_y.shape}')
                    
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.model == 'TimeDiff':
                    loss = self.model.train_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
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
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if self.args.model == 'TimeDiff':
                vali_loss = 1
                test_loss = 1
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
            wandb.log({"lr": scheduler.get_last_lr()[0],'epoch': epoch,"lr_optim": model_optim.param_groups[0]['lr']}, step=total_steps)
            wandb.log({"loss/train": train_loss, "loss/vali": vali_loss, "loss/test": test_loss}, step=total_steps)
            if self.args.dataset_name == 'taxi' and epoch%3!=0 and epoch!=self.args.train_epochs-1:
                print('skip test')
                # pass
            else:
                self.test(setting, save_results=False, total_steps=total_steps)
            total_steps += 1
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # best_model_path = path + '/' + 'checkpoint.pth'
        self.test(setting, save_results=True, total_steps=total_steps)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, test=0, save_results=False,total_steps=0, use_wandb=True,flod_name='setting_time'):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, f'{setting}_{self.args.time}', 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, f'{flod_name}', 'checkpoint.pth')))
        preds = []
        trues = []
        inputx = []
        folder_path = os.path.join(self.args.path_prefix, 'test_results', f'{setting}_{self.args.time}/')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # print(f'test batch {i}, batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}')
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    elif self.args.model == 'TimeDiff':
                        outputs,_,_,_,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if self.args.model != 'TimeDiff':
                    if i % 1 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        # for i in preds:
        #     print(i.shape)
        # for i in trues:
        #     print(i.shape)
        # for i in inputx:
        #     print(i.shape)
        # preds = np.array(preds)
        # trues = np.array(trues)
        # inputx = np.array(inputx)


        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        # n个 preds 纵向堆叠
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        preds_multisample = None
        if len(preds.shape) > 3:
            preds_multisample = copy.deepcopy(preds)
            preds = preds.mean(axis=1)
        # print(f'preds shape: {preds.shape}, trues shape: {trues.shape}, inputx shape: {inputx.shape}')

        # result save
        folder_path = os.path.join(self.args.path_prefix, 'results', f'{setting}_{self.args.time}/')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        metrics_all = {}
        nrmse_sum, nmae_sum, nrmse, nmae, mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print(f'nrmse_sum:{nrmse_sum}, nmae_sum:{nmae_sum}, nrmse:{nrmse}, nmae:{nmae}, mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, rse:{rse}, corr:{corr}')
        metrics_all.update({"metric/nrmse_sum": nrmse_sum, "metric/nmae_sum": nmae_sum, "metric/nrmse": nrmse, "metric/nmae": nmae, "metric/mae": mae, "metric/mse": mse, "metric/rmse": rmse, "metric/mape": mape, "metric/mspe": mspe, "metric/rse": rse})
        if use_wandb:
            wandb.log({"metric/nrmse_sum": nrmse_sum, "metric/nmae_sum": nmae_sum, "metric/nrmse": nrmse, "metric/nmae": nmae, "metric/mae": mae, "metric/mse": mse, "metric/rmse": rmse, "metric/mape": mape, "metric/mspe": mspe, "metric/rse": rse, "metric/corr": corr}, step=total_steps)
        
        CRPS, ND, NRMSE, CRPS_Sum, ND_Sum, NRMSE_Sum = GLUONTS_METRICS(preds, trues)
        print(f'CRPS:{CRPS}, ND:{ND}, NRMSE:{NRMSE}, CRPS_Sum:{CRPS_Sum}, ND_Sum:{ND_Sum}, NRMSE_Sum:{NRMSE_Sum}')
        metrics_all.update({"metric_glu/CRPS": CRPS, "metric_glu/ND": ND, "metric_glu/NRMSE": NRMSE, "metric_glu/CRPS_Sum": CRPS_Sum, "metric_glu/ND_Sum": ND_Sum, "metric_glu/NRMSE_Sum": NRMSE_Sum})
        if use_wandb:
            wandb.log({"metric_glu/CRPS": CRPS, "metric_glu/ND": ND, "metric_glu/NRMSE": NRMSE, "metric_glu/CRPS_Sum": CRPS_Sum, "metric_glu/ND_Sum": ND_Sum, "metric_glu/NRMSE_Sum": NRMSE_Sum}, step=total_steps)

        preds_rs = []
        trues_rs = []
        for pred in preds:
            pred = test_data.inverse_transform(pred)
            preds_rs.append(pred)
        for true in trues:
            true = test_data.inverse_transform(true)
            trues_rs.append(true)
        preds_rs = np.array(preds_rs)
        trues_rs = np.array(trues_rs)
        preds_rs = preds_rs.reshape(-1, preds_rs.shape[-2], preds_rs.shape[-1])
        trues_rs = trues_rs.reshape(-1, trues_rs.shape[-2], trues_rs.shape[-1])

        
        if preds_multisample is not None:
            n_sample = preds_multisample.shape[1]
            pred_len = preds_multisample.shape[2]
            n_features = preds_multisample.shape[3]
            preds_multisample_rs = [] 
            for pred_mult in preds_multisample: # preds_multisample [batch_size, n_sample, pred_len, n_features]
                pred_mult_rs = []
                for pred in pred_mult: # pred [pred_len, n_features]
                    pred = test_data.inverse_transform(pred)
                    pred_mult_rs.append(pred)
                pred_mult_rs = np.array(pred_mult_rs)
                preds_multisample_rs.append(pred_mult_rs)
            preds_multisample_rs = np.array(preds_multisample_rs)
            preds_multisample_rs = preds_multisample_rs.reshape(-1, n_sample, pred_len, n_features)
        nrmse_sum, nmae_sum, nrmse, nmae, mae, mse, rmse, mape, mspe, rse, corr = metric(preds_rs, trues_rs)
        print(f'nrmse_sum_rs:{nrmse_sum}, nmae_sum_rs:{nmae_sum}, nrmse_rs:{nrmse}, nmae_rs:{nmae}, mae_rs:{mae}, mse_rs:{mse}, rmse_rs:{rmse}, mape_rs:{mape}, mspe_rs:{mspe}, rse_rs:{rse}, corr_rs:{corr}')
        metrics_all.update({"metric_rs/nrmse_sum": nrmse_sum, "metric_rs/nmae_sum": nmae_sum, "metric_rs/nrmse": nrmse, "metric_rs/nmae": nmae, "metric_rs/mae": mae, "metric_rs/mse": mse, "metric_rs/rmse": rmse, "metric_rs/mape": mape, "metric_rs/mspe": mspe, "metric_rs/rse": rse})
        if use_wandb:
            wandb.log({"metric_rs/nrmse_sum": nrmse_sum, "metric_rs/nmae_sum": nmae_sum, "metric_rs/nrmse": nrmse, "metric_rs/nmae": nmae, "metric_rs/mae": mae, "metric_rs/mse": mse, "metric_rs/rmse": rmse, "metric_rs/mape": mape, "metric_rs/mspe": mspe, "metric_rs/rse": rse, "metric_rs/corr": corr}, step=total_steps)
        
        CRPS, ND, NRMSE, CRPS_Sum, ND_Sum, NRMSE_Sum = GLUONTS_METRICS(preds_rs, trues_rs)
        print(f'CRPS_rs:{CRPS}, ND_rs:{ND}, NRMSE_rs:{NRMSE}, CRPS_Sum_rs:{CRPS_Sum}, ND_Sum_rs:{ND_Sum}, NRMSE_Sum_rs:{NRMSE_Sum}')
        metrics_all.update({"metric_glu_rs/CRPS": CRPS, "metric_glu_rs/ND": ND, "metric_glu_rs/NRMSE": NRMSE, "metric_glu_rs/CRPS_Sum": CRPS_Sum, "metric_glu_rs/ND_Sum": ND_Sum, "metric_glu_rs/NRMSE_Sum": NRMSE_Sum})
        if use_wandb:
            wandb.log({"metric_glu_rs/CRPS": CRPS, "metric_glu_rs/ND": ND, "metric_glu_rs/NRMSE": NRMSE, "metric_glu_rs/CRPS_Sum": CRPS_Sum, "metric_glu_rs/ND_Sum": ND_Sum, "metric_glu_rs/NRMSE_Sum": NRMSE_Sum}, step=total_steps)

        if preds_multisample is not None:
            CRPS, ND, NRMSE, CRPS_Sum, ND_Sum, NRMSE_Sum = GLUONTS_METRICS(preds_multisample_rs, trues_rs)
            print(f'multisample_glu_rs/CRPS:{CRPS}, multisample_glu_rs/ND:{ND}, multisample_glu_rs/NRMSE:{NRMSE}, multisample_glu_rs/CRPS_Sum:{CRPS_Sum}, multisample_glu_rs/ND_Sum:{ND_Sum}, multisample_glu_rs/NRMSE_Sum:{NRMSE_Sum}')
            metrics_all.update({"multisample_glu_rs/CRPS": CRPS, "multisample_glu_rs/ND": ND, "multisample_glu_rs/NRMSE": NRMSE, "multisample_glu_rs/CRPS_Sum": CRPS_Sum, "multisample_glu_rs/ND_Sum": ND_Sum, "multisample_glu_rs/NRMSE_Sum": NRMSE_Sum})
            if use_wandb:
                wandb.log({"multisample_glu_rs/CRPS": CRPS, "multisample_glu_rs/ND": ND, "multisample_glu_rs/NRMSE": NRMSE, "multisample_glu_rs/CRPS_Sum": CRPS_Sum, "multisample_glu_rs/ND_Sum": ND_Sum, "multisample_glu_rs/NRMSE_Sum": NRMSE_Sum}, step=total_steps)
        if save_results:
            metrics_all.update({"dataset": self.args.dataset_name, "model": self.args.model, "train_epochs": self.args.train_epochs, "pred_len": self.args.pred_len, "label_len": self.args.label_len, "time": self.args.time, "setting": setting})
            f = open(os.path.join(self.args.path_prefix, 'results', 'results.txt'), 'a')
            f.write(setting + "  \n")
            f.write('nrmse:{}, nmae:{}, mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, corr:{}\n'.format(nrmse, nmae, mae, mse, rmse, mape, mspe, rse, corr))
            f.write('\n')
            f.write('\n')
            f.close()
            args_str = str(self.args).replace(',',' ')
            # save matrics to csv
            filename = os.path.join(self.args.path_prefix, 'results', f'{self.args.dataset_name}_metrics_{self.args.model}_all.csv')
            columns = list(metrics_all.keys())
            if not os.path.exists(filename):
                with open(filename, 'a') as f:
                    f.write(','.join(columns) + '\n')
            with open(filename, 'a') as f:
                f.write(','.join([str(metrics_all[col]) for col in columns]) + '\n')

            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
            np.save(folder_path + 'pred.npy', preds)
            # np.save(folder_path + 'true.npy', trues)
            # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, f'{setting}_{self.args.time}')
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = os.path.join(self.args.path_prefix, 'results', f'{setting}_{self.args.time}/')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
