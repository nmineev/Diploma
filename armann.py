import numpy as np
import scipy.optimize as optimize
from time import time

class ARMA_NN:
    def __init__(self, data, order, test_size):
        self.order = order
        self.ar_ord, self.ma_ord, self.nn_ar_ord, self.nn_ma_ord, self.nn_hid_size = order
        self.num_of_params = 1+self.ar_ord+self.ma_ord+(self.nn_ar_ord+self.nn_ma_ord+2)*self.nn_hid_size
        data = np.array(data)
        self.test_index = int(data.shape[0]*(1-test_size))
        self.train, self.test = data[:self.test_index], data[self.test_index:]
    
    @staticmethod
    def F(vect):
        logistic = lambda z: 1/(1+np.exp(-z))
        for i in range(vect.shape[0]):
            try:
                vect[i] = logistic(vect[i])
            except FloatingPointError:
                if vect[i]>0: vect[i] = 1
                else: vect[i] = 0
        return vect

    def derF(vect): return ARMA_NN.F(vect)-ARMA_NN.to_square(ARMA_NN.F(vect))
    
    def to_square(vect):
        for i in range(vect.shape[0]):
            try:
                vect[i] = vect[i]**2
            except FloatingPointError:
                vect[i] = np.inf
        return vect
    
    def RSS(W_arma, W1, W2, start_shocks, ar_ord, ma_ord, nn_ar_ord, nn_ma_ord, nn_hid_size, data, ret_pred_shocks=False):
        max_ar_ord, max_ma_ord = max(ar_ord, nn_ar_ord), max(ma_ord, nn_ma_ord)
        shocks = np.hstack((0, start_shocks))
        rss = 0
        delta1, delta2 = np.zeros((nn_hid_size)), 0
        grad_W_arma = np.zeros(1+ar_ord+ma_ord)
        grad_W1 = np.zeros((nn_ar_ord+nn_ma_ord+1, nn_hid_size))
        grad_W2 = np.zeros((nn_hid_size))
        if ret_pred_shocks:
            first_shocks = np.append(np.zeros((max_ar_ord-max_ma_ord if max_ar_ord-max_ma_ord > 0 else 0)), start_shocks[max_ma_ord-max_ar_ord if max_ma_ord-max_ar_ord>0 else 0:])
            pred_to_ret = np.append(data[:max_ar_ord]-first_shocks, np.zeros((data.shape[0]-max_ar_ord)))
            shocks_to_ret = np.append(first_shocks, np.zeros((data.shape[0]-max_ar_ord)))
        #-----loop-start---
        I_arma = np.hstack((1, data[max_ar_ord-1:None if max_ar_ord-ar_ord-1==-1 and max_ar_ord else max_ar_ord-ar_ord-1:-1], shocks[-1:-ma_ord-1:-1]))
        I = np.hstack((1, data[max_ar_ord-1:None if max_ar_ord-nn_ar_ord-1==-1 and max_ar_ord else max_ar_ord-nn_ar_ord-1:-1], shocks[-1:-nn_ma_ord-1:-1]))
        A1 = np.dot(I, W1)
        Z2 = ARMA_NN.F(A1)
        pred = np.dot(I_arma, W_arma) + np.dot(Z2, W2)
        shocks = np.append(shocks[1:], data[max_ar_ord]-pred)
        delta2 = -shocks[-1]
        delta1 = ARMA_NN.derF(A1)*W2*delta2
        grad_W_arma += I_arma*delta2
        grad_W1 += np.tensordot(I, delta1, axes=0)
        grad_W2 += Z2*delta2
        if ret_pred_shocks:
            pred_to_ret[max_ar_ord] = pred
            shocks_to_ret[max_ar_ord] = shocks[-1]
        try:
            rss += shocks[-1]**2
        except FloatingPointError:
            rss = np.inf
        for i in range(max_ar_ord+1, data.shape[0]):
            I_arma = np.hstack((1, data[i-1:i-ar_ord-1:-1], shocks[-1:-ma_ord-1:-1]))
            I = np.hstack((1, data[i-1:i-nn_ar_ord-1:-1], shocks[-1:-nn_ma_ord-1:-1]))
            A1 = np.dot(I, W1)
            Z2 = ARMA_NN.F(A1)
            pred = np.dot(I_arma, W_arma) + np.dot(Z2, W2)
            shocks = np.append(shocks[1:], data[i]-pred)
            delta2 = -shocks[-1]
            delta1 = ARMA_NN.derF(A1)*W2*delta2
            grad_W_arma += I_arma*delta2
            grad_W1 += np.tensordot(I, delta1, axes=0)
            grad_W2 += Z2*delta2
            if ret_pred_shocks:
                pred_to_ret[i] = pred
                shocks_to_ret[i] = shocks[-1]
            if rss != np.inf:
                try:
                    rss += shocks[-1]**2
                except FloatingPointError:
                    rss = np.inf
        #------------
        if ret_pred_shocks: return pred_to_ret, shocks_to_ret
        return 0.5*(rss), np.hstack((grad_W_arma, grad_W1.ravel(), grad_W2))
    
    def RSS_to_opt(params, ar_ord, ma_ord, nn_ar_ord, nn_ma_ord, nn_hid_size, data, exact):
        W_arma = params[:1+ar_ord+ma_ord]
        W1 = params[1+ar_ord+ma_ord:1+ar_ord+ma_ord+(nn_ar_ord+nn_ma_ord+1)*nn_hid_size].reshape((nn_ar_ord+nn_ma_ord+1, nn_hid_size))
        if exact:
            W2 = params[1+ar_ord+ma_ord+(nn_ar_ord+nn_ma_ord+1)*nn_hid_size:1+ar_ord+ma_ord+(1+nn_ar_ord+nn_ma_ord)*nn_hid_size+nn_hid_size]
            start_shocks = params[1+ar_ord+ma_ord+(1+nn_ar_ord+nn_ma_ord)*nn_hid_size+nn_hid_size:]
        else:
            W2 = params[-nn_hid_size:]
            start_shocks = np.zeros((max(ma_ord, nn_ma_ord)))
        if exact: return ARMA_NN.RSS(W_arma, W1, W2, start_shocks, ar_ord, ma_ord, nn_ar_ord, nn_ma_ord, nn_hid_size, data)[0]
        return ARMA_NN.RSS(W_arma, W1, W2, start_shocks, ar_ord, ma_ord, nn_ar_ord, nn_ma_ord, nn_hid_size, data)

    def optres_unpack(params, ar_ord, ma_ord, nn_ar_ord, nn_ma_ord, nn_hid_size, exact):
        W_arma = params[:1+ar_ord+ma_ord]
        W1 = params[1+ar_ord+ma_ord:1+ar_ord+ma_ord+(1+nn_ar_ord+nn_ma_ord)*nn_hid_size].reshape((nn_ar_ord+nn_ma_ord+1, nn_hid_size))
        if exact:
            W2 = params[1+ar_ord+ma_ord+(1+nn_ar_ord+nn_ma_ord)*nn_hid_size:1+ar_ord+ma_ord+(1+nn_ar_ord+nn_ma_ord)*nn_hid_size+nn_hid_size]
            start_shocks = params[1+ar_ord+ma_ord+(1+nn_ar_ord+nn_ma_ord)*nn_hid_size+nn_hid_size:]
        else:
            W2 = params[-nn_hid_size:]
            start_shocks = np.zeros((max(ma_ord, nn_ma_ord)))
        return W_arma, W1, W2, start_shocks   

    def fit(self, rand_steps=3, solver="l-bfgs-b", maxiter=500, maxfun=15000, tol=1e-4, iprint=0, exact=True, jac=True, rand_init=True):
        tik = time()
        #--optimization---
        rss_min, count, best_optres = np.inf, 0, None
        while count < rand_steps:
            try:
                if exact:
                    if rand_init: init = np.random.normal(size=self.num_of_params+max(self.ma_ord, self.nn_ma_ord))
                    else: init = np.zeros((self.num_of_params+max(self.ma_ord, self.nn_ma_ord)))
                    optres = optimize.minimize(ARMA_NN.RSS_to_opt, init, (self.ar_ord, self.ma_ord, self.nn_ar_ord, self.nn_ma_ord, self.nn_hid_size, self.train, exact), method=solver, jac=jac, options={"gtol":tol, "maxfun":maxfun, "maxiter":maxiter, "iprint":iprint})
                else:
                    if rand_init: init = np.random.normal(size=self.num_of_params)
                    else: init = np.zeros((self.num_of_params))
                    optres = optimize.minimize(ARMA_NN.RSS_to_opt, init, (self.ar_ord, self.ma_ord, self.nn_ar_ord, self.nn_ma_ord, self.nn_hid_size, self.train, exact), method=solver, jac=jac, options={"gtol":tol, "maxfun":maxfun, "maxiter":maxiter, "iprint":iprint})           
            except FloatingPointError:
                print("Get Error on init:{}".format(init))
                continue
            if optres.fun == np.inf: 
                print("Diverge on init:{}".format(init))
                continue
            count+=1
            if optres.fun < rss_min:
                rss_min = optres.fun
                best_optres = optres
                print("New best result:{}".format(best_optres.fun))
                print("on init:{}".format(init))
        #------------
        print("fit-time:{}".format(time()-tik))
        self.W_arma, self.W1, self.W2, self.start_shocks = ARMA_NN.optres_unpack(best_optres.x, self.ar_ord, self.ma_ord, self.nn_ar_ord, self.nn_ma_ord, self.nn_hid_size, exact)
        self.pred, self.shocks = ARMA_NN.RSS(self.W_arma, self.W1, self.W2, self.start_shocks, self.ar_ord, self.ma_ord, self.nn_ar_ord, self.nn_ma_ord, self.nn_hid_size, self.train, ret_pred_shocks=True)
        self.std_dev = np.sqrt(np.sum(np.square(self.shocks-np.mean(self.shocks)))/(self.shocks.shape[0]-1))
        self.loglik = -0.5*(self.train.shape[0]-max(self.ar_ord, self.nn_ar_ord))*np.log(2*np.pi*self.std_dev**2)-best_optres.fun/self.std_dev**2
        self.aic = -2/self.train.shape[0]*(self.loglik-self.num_of_params)
        self.mse = np.sum(np.square(self.shocks[self.ar_ord:]))/self.train.shape[0]
        self.rmse = np.sqrt(self.mse)
        return self 

    def predict(self, test=None):
        if test is None: test = self.test
        predicted = np.zeros((test.shape[0]))
        shocks = np.append(self.shocks, np.zeros((test.shape[0])))
        m = self.train.shape[0]
        data = np.append(self.train, test)
        for i in range(m, m+test.shape[0]):
            I_arma = np.hstack((1, data[i-1:i-self.ar_ord-1:-1], shocks[i-1:i-self.ma_ord-1:-1]))
            I = np.hstack((1, data[i-1:i-self.nn_ar_ord-1:-1], shocks[i-1:i-self.nn_ma_ord-1:-1]))
            predicted[i-m] = np.dot(I_arma, self.W_arma) + np.dot(ARMA_NN.F(np.dot(I, self.W1)), self.W2)
            shocks[i] = test[i-m]-predicted[i-m]
        self.test_mse = np.sum(np.square(shocks[m:]))/data.shape[0]
        self.test_rmse = np.sqrt(self.test_mse)
        return predicted
    
    def forecast(self, n_pred): pass