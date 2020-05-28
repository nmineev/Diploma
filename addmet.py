import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product

import arma
import nn
import armann
import armann_zhang

#Additional Methods

def inv_diff(data, diff_data, diff_ord=1):
    if not diff_ord: return diff_data
    act_diff_data = np.diff(data, n=diff_ord-1)
    return inv_diff(data, np.append(act_diff_data[0], act_diff_data[:-1]+diff_data), diff_ord-1)

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_true-y_pred)))

def tsplot(y, lags=None, figsize=(10, 8), style='bmh', max_lag=10):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        dful_pvalue = np.around(smt.stattools.adfuller(y)[1], 3)
        ACF = smt.stattools.acf(y, nlags=max_lag, qstat=True)
        ARord = np.array([i for i in range(0, max_lag+1) if abs(ACF[0][i])>2/np.sqrt(y.shape[0])])
        PACF = smt.stattools.pacf(y, nlags=max_lag)
        MAord = np.array([i for i in range(0, max_lag+1) if abs(PACF[i])>2/np.sqrt(y.shape[0])])
        Qstat_pvalue = np.around(ACF[2][max_lag-1], 3)
        jb_pvalue = sm.stats.stattools.jarque_bera(y)
        jb_pvalue, kurtosis = np.around(jb_pvalue[1], 3), np.around(jb_pvalue[3], 3)

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots\nDickey-Fuller Test: {}'.format(dful_pvalue))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        qq_ax.set_title('QQ Plot\nJarque-Bera Test: {}\nKurtosis: {}'.format(jb_pvalue, kurtosis))
        acf_ax.set_title("Autocorrelation\nQ({}): {}\nLast Singf Lag: {}".format(max_lag, Qstat_pvalue, max(ARord)))
        pacf_ax.set_title("Partial Autocorrelation\nLast Singf Lag: {}".format(max(MAord)))

        plt.tight_layout()
    plt.show()
    return ARord, MAord

def plotResult(data, mdl, test_index=0, ind1=0, ind2=None, diff_ord=1, stattools=False):
    data = np.array(data)
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,15))
    if not stattools: pred = inv_diff(data, np.append(mdl.pred, mdl.predict()), diff_ord)
    else: pred = inv_diff(data, mdl.fittedvalues, diff_ord)
    ax1.plot(pred[ind1:ind2], label = "Model")
    ax1.plot(data[ind1:ind2], label = "Actual")
    error = mean_absolute_percentage_error(data[ind1:ind2], pred[ind1:ind2])
    ax1.set_title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    if not (ind1 or ind2): ax1.axvspan(len(data)-len(data[test_index:]), len(data), alpha=0.3, color='lightgrey')
    ax1.grid(True)
    ax1.axis('tight')
    ax1.legend(loc="best", fontsize=13);
    
    ax2.plot(pred[test_index:], label = "Model")
    ax2.plot(data[test_index:], label = "Actual")
    error = mean_absolute_percentage_error(data[test_index:], pred[test_index:])
    ax2.set_title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    ax2.grid(True)
    ax2.axis('tight')
    ax2.legend(loc="best", fontsize=13);
    plt.show()

def ResultTable(data, MODELS, test_index=0, index=None, diff_ord=1):
    data = np.array(data)
    RES = pd.DataFrame(index=index)
    rmse_vals = np.zeros((len(MODELS)))
    loglik_vals = np.zeros((len(MODELS)))
    mape_vals = np.zeros((len(MODELS)))
    for i in range(len(MODELS)):
        pred = inv_diff(data, np.append(MODELS[i].pred, MODELS[i].predict()), diff_ord)
        rmse_vals[i] = root_mean_squared_error(data[test_index:], pred[test_index:])
        mape_vals[i] = mean_absolute_percentage_error(data[test_index:], pred[test_index:])
        loglik_vals[i] = MODELS[i].loglik
    RES["rmse"] = rmse_vals
    RES["mape"] = mape_vals
    RES["loglik"] = loglik_vals
    return RES

def get_valid_params(max_ar_ord, max_ma_ord, max_in_size, max_hid_size, model_type):
    ar_params, ma_params, in_sizes, hid_sizes = np.arange(max_ar_ord+1), np.arange(max_ma_ord+1), np.arange(max_in_size+1), np.arange(1, max_hid_size+1)
    if model_type.lower() == "arma":
        params_list = list(product(ar_params, ma_params)).remove((0, 0))
    elif model_type.lower() == "nn":
        params_list = list(product(ar_params, ma_params, hid_sizes))
        params_list_copy = params_list.copy()
        for el in params_list_copy:
            if not (el[0] or el[1]): params_list.remove(el)
            elif (el[0]+el[1])>el[2]: params_list.remove(el)
    elif model_type.lower() == "armann":
        params_list = list(product(ar_params, ma_params, ar_params, ma_params, hid_sizes))
        params_list_copy = params_list.copy()
        for el in params_list_copy:
            if not (el[0] or el[1] or el[2] or el[3]): params_list.remove(el)
            elif (el[2]+el[3])>el[4]: params_list.remove(el)
            elif not (el[0] or el[1]): params_list.remove(el)
            elif not (el[2] or el[3]): params_list.remove(el)
    elif model_type.lower() == "armann_zhang":
        params_list = list(product(ar_params, ma_params, in_sizes, hid_sizes))
        params_list_copy = params_list.copy()
        for el in params_list_copy:
            if not (el[0] or el[1]): params_list.remove(el)
            elif not (el[2] and el[3]): params_list.remove(el)
            elif el[2]>el[3]: params_list.remove(el)
    else: 
        print("Invalid type")
        return None
    return params_list

def AICOptimizer(model_type, max_ar_ord, max_ma_ord, max_hid_size, data, params_list=None, insurance=200, 
                 rand_steps=3, solver="l-bfgs-b", maxiter=500, maxfun=15000, tol=1e-4, iprint=0, 
                 exact=True, jac=True, rand_init=True):
    if params_list is None: params_list = get_valid_params(max_ar_ord, max_ma_ord, max_hid_size, model_type)
    aic_min, count, best_model, best_order = np.inf, 0, None, None
    RES = pd.DataFrame(index=["order", "aic"])
    ERRs = pd.DataFrame(index=["order"])
    for order in params_list:
        try:
            if model_type.lower() == "arma":
                model = arma.ARMA(data, order, 0).fit(rand_steps=rand_steps, solver=solver, maxiter=maxiter, maxfun=maxfun, tol=tol, iprint=iprint, 
                                 exact=exact, jac=jac, rand_init=rand_init)
            elif model_type.lower() == "nn": 
                model = nn.NN(data, order, 0).fit(rand_steps=rand_steps, solver=solver, maxiter=maxiter, maxfun=maxfun, tol=tol, iprint=iprint, 
                             exact=exact, jac=jac, rand_init=rand_init)
            elif model_type.lower() == "armann": 
                model = armann.ARMA_NN(data, order, 0).fit(rand_steps=rand_steps, solver=solver, maxiter=maxiter, maxfun=maxfun, tol=tol, iprint=iprint, 
                                      exact=exact, jac=jac, rand_init=rand_init)
            elif model_type.lower() == "armann_zhang": 
                model = armann_zhang.ARMA_NN_Zhang(data, order, 0).fit(rand_steps=rand_steps, solver=solver, maxiter=maxiter, maxfun=maxfun, tol=tol, iprint=iprint, 
                                      exact=exact, jac=jac, rand_init=rand_init)
            else: 
                print("Invalid type")
                return None
        except:
            print("||ERROR|| Error on order:{}".format(order))
            ERRs = ERRs.append({"order": order})
            continue
        if model.aic < aic_min:
            aic_min = model.aic
            best_model = model
            best_order = order
        RES = RES.append({"order": model.order, "aic": model.aic}, ignore_index=True)
        print("||New Model Fit|| model order:{}, model aic:{}".format(model.order, model.aic))
        count+=1
        if count == insurance: 
            RES.to_excel("models_aic.xlsx")
            count = 0
    return best_model, best_order, aic_min, RES, ERRs