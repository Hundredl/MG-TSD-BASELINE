import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import wandb
import datetime
wandb.login(key='94cc33acc0d4f4be3396e28131e30d1f057d3487')
# 设置环境变量WANDB_MODE=dryrun
# os.environ['WANDB_MODE'] = 'dryrun'
# wandb offline
if __name__ == '__main__':
    print('Start running...')
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument('--exp_label', type=str, default='LongForecasting', help='experiment label')
    parser.add_argument('--blob_prefix', type=str, default='/home/v-wuyueying/workspace/blob/drive/')
    parser.add_argument('--path_prefix', type=str, default='mgtsd/baseline/patchtst/')
    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='checkpoints/', help='location of model checkpoints')
    parser.add_argument('--dataset_name', type=str, default='ETTm1', help='dataset name')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # Diffusion Models
    parser.add_argument('--interval', type=int, default=100, help='number of diffusion steps')
    parser.add_argument('--ot-ode', default=True, help='use OT-ODE model')
    parser.add_argument("--beta-max", type=float, default=0.3, help="max diffusion for the diffusion model")
    parser.add_argument("--t0", type=float, default=1e-4, help="sigma start time in network parametrization")
    parser.add_argument("--T", type=float, default=1., help="sigma end time in network parametrization")
    parser.add_argument('--model_channels', type=int, default=256)
    parser.add_argument('--nfe', type=int, default=100)
    parser.add_argument('--dim_LSTM', type=int, default=64)
    parser.add_argument('--num_vars', type=int, default=7, help='encoder input size')
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='train epochs')


    parser.add_argument('--diff_steps', type=int, default=100, help='number of diffusion steps')
    parser.add_argument('--UNet_Type', type=str, default='CNN', help=['CNN'])
    parser.add_argument('--D3PM_kernel_size', type=int, default=5)
    parser.add_argument('--use_freq_enhance', type=int, default=0)
    parser.add_argument('--type_sampler', type=str, default='dpm', help=["none", "dpm"])
    parser.add_argument('--parameterization', type=str, default='x_start', help=["noise", "x_start"])

    parser.add_argument('--ddpm_inp_embed', type=int, default=256)
    parser.add_argument('--ddpm_dim_diff_steps', type=int, default=100)
    parser.add_argument('--ddpm_channels_conv', type=int, default=256)
    parser.add_argument('--ddpm_channels_fusion_I', type=int, default=256)
    parser.add_argument('--ddpm_layers_inp', type=int, default=5)
    parser.add_argument('--ddpm_layers_I', type=int, default=5)
    parser.add_argument('--ddpm_layers_II', type=int, default=5)
    parser.add_argument('--cond_ddpm_num_layers', type=int, default=5)
    parser.add_argument('--cond_ddpm_channels_conv', type=int, default=64)

    parser.add_argument('--ablation_study_case', type=str, default="none", help="none, mix_1, ar_1, mix_ar_0, w_pred_loss")
    parser.add_argument('--weight_pred_loss', type=float, default=0.0)
    parser.add_argument('--ablation_study_F_type', type=str, default="CNN", help="Linear, CNN")
    parser.add_argument('--ablation_study_masking_type', type=str, default="none", help="none, hard, segment")
    parser.add_argument('--ablation_study_masking_tau', type=float, default=0.9)

    parser.add_argument('--vis_ar_part', type=int, default=0, help='status')
    parser.add_argument('--use_window_normalization', type=bool, default=True)
    parser.add_argument('--sample_times', type=int, default=10)
    






    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    args.blob_prefix = os.environ['AMLT_BLOB_ROOT_DIR'] if 'AMLT_BLOB_ROOT_DIR' in os.environ else args.blob_prefix
    args.path_prefix = os.path.join(args.blob_prefix, args.path_prefix)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('os.path.join(args.path_prefix, args.root_path): ', os.path.join(args.path_prefix, args.root_path))
    print('os.path.join(args.path_prefix, args.checkpoints): ', os.path.join(args.path_prefix, args.checkpoints))
    args.root_path = os.path.join(args.path_prefix, args.root_path)
    args.checkpoints = os.path.join(args.path_prefix, args.checkpoints)
    print('args.root_path: ', args.root_path)
    print('args.checkpoints: ', args.checkpoints)
    input_dim_dict = {
        'elec':370,'sol':137,'cup':270,'traf':963,'taxi':1214,'wiki':2000}
    args.input_dim = input_dim_dict[args.dataset_name]
    args.enc_in = input_dim_dict[args.dataset_name]
    args.num_vars = input_dim_dict[args.dataset_name]
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d_%H-%M-%S')
    exp_name = f'patchtst_{now}'
    args.time = now
    
    Exp = Exp_Main

    if args.is_training:
        wandb.init(project='PatchTST_Glounts', config=args, name=exp_name)
        for ii in range(args.itr):
            # setting record of experiments

            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.dataset_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,ii)

            exp = Exp(args)  # set experiments
            if args.model == 'TimeDiff':
                print('>>>>>>>start pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.pretrain(setting)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting,save_results=True, save_results_path='./results/', total_steps=None)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.dataset_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,ii)
        times = {
            'elec':['2023-11-17_02-37-23','2023-11-17_02-37-22','2023-11-17_02-37-21','2023-11-17_02-14-52','2023-11-17_02-13-28',
                    '2023-11-16_12-49-21','2023-11-16_12-49-20','2023-11-16_18-57-04','2023-11-16_08-54-05','2023-11-16_08-53-54'],
            'sol':['2023-11-17_02-37-24','2023-11-17_02-37-25','2023-11-17_02-37-23','2023-11-17_02-14-55','2023-11-17_02-11-32'],
            'cup':['2023-11-17_02-38-57','2023-11-17_02-37-23','2023-11-17_02-37-23','2023-11-17_02-14-43','2023-11-17_02-12-43'],
            'taxi':['2023-11-17_02-38-55','2023-11-17_02-37-25','2023-11-17_02-13-29','2023-11-17_02-13-21','2023-11-17_02-12-20'],
            # 'traf':['2023-11-17_02-38-57','2023-11-17_02-37-24','2023-11-17_02-37-22','2023-11-17_02-13-44','2023-11-17_02-10-23'],
            'traf':['2023-11-17_02-10-23'],
            'wiki':['2023-11-17_02-37-26','2023-11-17_02-37-24','2023-11-17_02-37-25','2023-11-17_02-11-18','2023-11-17_02-06-05'],
            
        }
        for cur_time in times[args.dataset_name]:
            flod_name = f'{setting}_{cur_time}'
            exp = Exp(args)
            # flod_name = f'{setting}_2023-11-17_02-37-24'
            # exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1, save_results=True, total_steps=0, flod_name=flod_name, use_wandb=False)
            torch.cuda.empty_cache()
        