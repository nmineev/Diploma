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

def plot_model_result(data, mdl, label_model="Model", label_data="Actual", xlabel="x", ylabel="y", color=None, data_index=None, path_to_save=None, test_index=0, ind1=0, ind2=None, figsize=(15, 15), diff_ord=1):
    data = np.array(data)
    if data_index is None: data_index = np.arange(data.shape[0])
    
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    pred = inv_diff(data, np.append(mdl.pred, mdl.predict()), diff_ord)
    ax1.plot(data_index[ind1:ind2], pred[ind1:ind2], color="m", label = label_model)
    ax1.plot(data_index[ind1:ind2], data[ind1:ind2], color="c", label = label_data)
    error = mean_absolute_percentage_error(data[ind1:ind2], pred[ind1:ind2])
    if not (ind1 or ind2): ax1.axvspan(data_index[test_index], data_index[-1], alpha=0.3, color='lightgrey')
        
    if color is None:
        ax1.set_title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        
        ax1.grid(True)
    else:
        ax1.set_title("Mean Absolute Percentage Error: {0:.2f}%".format(error), color=color)
        
        ax1.set_xlabel(xlabel, color=color)
        ax1.set_ylabel(ylabel, color=color)

        ax1.spines['bottom'].set_color(color)
        ax1.spines['top'].set_color(color)
        ax1.spines['right'].set_color(color)
        ax1.spines['left'].set_color(color)

        ax1.tick_params(axis='x', colors=color)
        ax1.tick_params(axis='y', colors=color)
        
        ax1.grid(True, color=color)
    
    ax1.axis('tight')
    ax1.legend(loc="best", fontsize=13);
    
    ax2.plot(data_index[test_index:], pred[test_index:], color="m", label = label_model)
    ax2.plot(data_index[test_index:], data[test_index:], color="c", label = label_data)
    error = mean_absolute_percentage_error(data[test_index:], pred[test_index:])
    
    if color is None:
        ax2.set_title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        
        ax2.grid(True)
    else:
        ax2.set_title("Mean Absolute Percentage Error: {0:.2f}%".format(error), color=color)
        
        ax2.set_xlabel(xlabel, color=color)
        ax2.set_ylabel(ylabel, color=color)

        ax2.spines['bottom'].set_color(color)
        ax2.spines['top'].set_color(color)
        ax2.spines['right'].set_color(color)
        ax2.spines['left'].set_color(color)

        ax2.tick_params(axis='x', colors=color)
        ax2.tick_params(axis='y', colors=color)
        
        ax2.grid(True, color=color)
    
    ax2.axis('tight')
    ax2.legend(loc="best", fontsize=13);
    if path_to_save is not None: plt.savefig(path_to_save, transparent=True, dpi=100)
    plt.show()

def plot_result(data, MDLS, LABELS=None, label_data="Actual", xlabel="x", ylabel="y", COLORS=None, color=None, data_index=None, path_to_save=None, test_index=0, figsize=(15, 7), diff_ord=1):
    data = np.array(data)
    error_min, indx = np.inf, None
    if data_index is None: data_index = np.arange(data.shape[0])
    if LABELS is None: LABELS = [str(mdl.order) for mdl in MDLS]
    
    f, ax = plt.subplots(figsize=figsize)
    ax.plot(data_index[test_index:], data[test_index:], color="c", label = label_data)
    for i in range(len(MDLS)):
        pred = inv_diff(data, np.append(MDLS[i].pred, MDLS[i].predict()), diff_ord)
        if COLORS is not None:
            ax.plot(data_index[test_index:], pred[test_index:], color=COLORS[i], label=LABELS[i])
        else:
            ax.plot(data_index[test_index:], pred[test_index:], label=LABELS[i])
        error = mean_absolute_percentage_error(data[test_index:], pred[test_index:])
        if error < error_min:
            error_min = error
            indx = i
    
    if color is None:
        ax.set_title("Least MAPE: {0:.2f}% on {1:}".format(error, LABELS[indx]))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        ax.grid(True)
    else:
        ax.set_title("Least MAPE: {0:.2f}% on {1:}".format(error, LABELS[indx]), color=color)
        
        ax.set_xlabel(xlabel, color=color)
        ax.set_ylabel(ylabel, color=color)

        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)

        ax.tick_params(axis='x', colors=color)
        ax.tick_params(axis='y', colors=color)
        
        ax.grid(True, color=color)
    
    ax.axis('tight')
    ax.legend(loc="best", fontsize=13);
    if path_to_save is not None: plt.savefig(path_to_save, transparent=True, dpi=100)
    plt.show()

def result_table(data, MODELS, test_index=0, LABELS=None, diff_ord=1):
    data = np.array(data)
    RES = pd.DataFrame(index=LABELS)
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

def get_valid_params(max_ar_ord, max_ma_ord, max_in_size, max_hid_size, model_type, old_params_list=None):
    ar_params, ma_params, in_sizes, hid_sizes = np.arange(max_ar_ord+1), np.arange(max_ma_ord+1), np.arange(max_in_size+1), np.arange(1, max_hid_size+1)
    if model_type.lower() == "arma":
        params_list = list(product(ar_params, ma_params)).remove((0, 0))
        if old_params_list is not None:
            for el in old_params_list: params_list.remove(el)
    elif model_type.lower() == "nn":
        params_list = list(product(ar_params, ma_params, hid_sizes))
        if old_params_list is not None:
            for el in old_params_list: params_list.remove(el)
        params_list_copy = params_list.copy()
        for el in params_list_copy:
            if not (el[0] or el[1]): params_list.remove(el)
            elif (el[0]+el[1])>el[2]: params_list.remove(el)
    elif model_type.lower() == "armann":
        params_list = list(product(ar_params, ma_params, ar_params, ma_params, hid_sizes))
        if old_params_list is not None:
            for el in old_params_list: params_list.remove(el)
        params_list_copy = params_list.copy()
        for el in params_list_copy:
            if not (el[0] or el[1] or el[2] or el[3]): params_list.remove(el)
            elif (el[2]+el[3])>el[4]: params_list.remove(el)
            elif not (el[0] or el[1]): params_list.remove(el)
            elif not (el[2] or el[3]): params_list.remove(el)
    elif model_type.lower() == "armann_zhang":
        params_list = list(product(ar_params, ma_params, in_sizes, hid_sizes))
        if old_params_list is not None:
            for el in old_params_list: params_list.remove(el)
        params_list_copy = params_list.copy()
        for el in params_list_copy:
            if not (el[0] or el[1]): params_list.remove(el)
            elif not (el[2] and el[3]): params_list.remove(el)
            elif el[2]>el[3]: params_list.remove(el)
    else: 
        print("Invalid model type")
        return None
    return params_list

def AICOptimizer(model_type, max_ar_ord, max_ma_ord, max_in_size, max_hid_size, data, old_params_list=None, insurance=200, 
                 rand_steps=3, solver="l-bfgs-b", maxiter=500, maxfun=15000, tol=1e-4, iprint=0, 
                 exact=True, jac=True, rand_init=True):
    params_list = get_valid_params(max_ar_ord, max_ma_ord, max_in_size, max_hid_size, model_type, old_params_list)
    aic_min, insurance_count, best_model, best_order = np.inf, 0, None, None
    RES = pd.DataFrame(index=["order", "aic"])
    ERRs = []
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
                print("Invalid model type")
                return None
        except:
            print("||ERROR|| Error on order:{}".format(order))
            ERRs.append(order)
            continue
        if model.aic < aic_min:
            aic_min = model.aic
            best_model = model
            best_order = order
        RES = RES.append({"order": model.order, "aic": model.aic}, ignore_index=True)
        print("||New Model Fit|| model order:{}, model aic:{}".format(model.order, model.aic))
        insurance_count+=1
        if insurance_count == insurance: 
            RES.to_excel("models_aic.xlsx")
            insurance_count = 0
    return best_model, best_order, aic_min, RES, ERRs