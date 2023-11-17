import numpy as np
import pandas as pd
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def NRMSE(pred, true):
    # return np.sqrt(MSE(pred, true)/np.mean(np.abs(true)))
    return np.sqrt(MSE(pred, true))/np.mean(np.abs(true))

def NMAE(pred, true):
    return np.mean(np.abs(pred - true))/np.mean(np.abs(true))

def NRMSE_SUM(pred, true):
    pred = pred.sum(axis=-1)
    true = true.sum(axis=-1)
    return np.sqrt(MSE(pred, true))/np.sum(np.abs(true))

def NMAE_SUM(pred, true):
    pred = pred.sum(axis=-1)
    true = true.sum(axis=-1)
    return np.mean(np.abs(pred - true))/np.sum(np.abs(true))

def GLUONTS_METRICS(pred, true):
    '''
    get metrics from gluonts

    Parameters
    ----------
    pred : np.array [num_samples, pred_length, num_series] 7*24*370
    true : np.array [num_samples, pred_length, num_series]
    '''
    # evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0).tolist[1:],
                                    #   target_agg_funcs={'sum': np.sum})

    # change pred to Forcast object
    from gluonts.model.forecast import SampleForecast
    pred_list = []
    for i in range(len(pred)):
        # extend the dimension of pred[i] from [pred_length, num_series] to [1, pred_length, num_series]
        if len(pred[i].shape) == 2:
            pred_cur = np.expand_dims(pred[i], axis=0)
        else:
            pred_cur = pred[i]
        # get the start date of the current pred
        start_date = pd.Period('2012-01-01 00:00:00', freq='H') + pd.to_timedelta(i * len(pred[i]), unit='H')
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        # create a Forcast object
        pred_forecast = SampleForecast(samples=pred_cur, start_date=pd.Period(start_date, freq='H'))
        pred_list.append(pred_forecast)
    
    true_list = []
    for i in range(len(true)):
        start_date = pd.Period('2012-01-01 00:00:00', freq='H') + pd.to_timedelta(i * len(true[i]), unit='H')
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        true_list.append(pd.DataFrame(data=true[i], index=pred_list[i].index)) # copy index from pred_list[i]

    evaluator = MultivariateEvaluator(
                                    # quantiles=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95],
                                    quantiles=(np.arange(20) / 20.0)[1:],
                                    target_agg_funcs={'sum': np.sum})
    agg_metric, item_metric = evaluator(true_list, pred_list, num_series=true.shape[-1])

    
    CRPS = agg_metric["mean_wQuantileLoss"]
    ND = agg_metric["ND"]
    NRMSE = agg_metric["NRMSE"]
    CRPS_Sum = agg_metric["m_sum_mean_wQuantileLoss"]
    ND_Sum = agg_metric["m_sum_ND"]
    NRMSE_Sum = agg_metric["m_sum_NRMSE"]
    return CRPS, ND, NRMSE, CRPS_Sum, ND_Sum, NRMSE_Sum

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    nrmse = NRMSE(pred, true)
    nmae = NMAE(pred, true)
    nrmse_sum = NRMSE_SUM(pred, true)
    nmae_sum = NMAE_SUM(pred, true)
    return nrmse_sum, nmae_sum, nrmse, nmae, mae, mse, rmse, mape, mspe, rse, corr