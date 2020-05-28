import numpy as np
import scipy.optimize as optimize
from time import time

class ARMA:
    def __init__(self, data, order, test_size):
        self.order = order
        self.ar_ord, self.ma_ord = order
        self.num_of_params = 1+self.ar_ord+self.ma_ord
        data = np.array(data)
        self.test_index = int(data.shape[0]*(1-test_size))
        self.train, self.test = data[:self.test_index], data[self.test_index:]
    
    def vect_on_scal(vect, scal):
        for i in range(vect.shape[0]):
            try:
                vect[i] = vect[i]*scal
            except FloatingPointError:
                vect[i] = np.sign(scal)*np.sign(vect[i])*np.inf
        return vect
    
    @staticmethod
    def RSS(W_arma, start_shocks, ar_ord, ma_ord, data, ret_pred_shocks=False):
        shocks = np.hstack((0, start_shocks))
        delta = 0
        rss = 0
        grad_W_arma = np.zeros((W_arma.shape[0]))
        if ret_pred_shocks:
            first_shocks = np.append(np.zeros((ar_ord-ma_ord if ar_ord-ma_ord > 0 else 0)),
                                     start_shocks[ma_ord-ar_ord if ma_ord-ar_ord>0 else 0:])
            pred_to_ret = np.append(data[:ar_ord]-first_shocks, 
                                    np.zeros((data.shape[0]-ar_ord)))
            shocks_to_ret = np.append(first_shocks, np.zeros((data.shape[0]-ar_ord)))
        #-----loop-start---
        arma_in = np.hstack((1, data[ar_ord-1:None if ar_ord else -1:-1], 
                             shocks[-1:-ma_ord-1:-1]))
        pred = np.dot(arma_in, W_arma)
        shocks = np.append(shocks[1:], data[ar_ord]-pred)
        delta = -shocks[-1]
        grad_W_arma += ARMA.vect_on_scal(arma_in, delta)
        if ret_pred_shocks:
            pred_to_ret[ar_ord] = pred
            shocks_to_ret[ar_ord] = shocks[-1]
        try:
            rss += shocks[-1]**2
        except FloatingPointError:
            rss = np.inf
        
        for i in range(ar_ord+1, data.shape[0]):
            arma_in = np.hstack((1, data[i-1:i-ar_ord-1:-1], shocks[-1:-ma_ord-1:-1]))
            pred = np.dot(arma_in, W_arma)
            shocks = np.append(shocks[1:], data[i]-pred)
            delta = -shocks[-1]
            grad_W_arma += ARMA.vect_on_scal(arma_in, delta)
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
        return 0.5*(rss), grad_W_arma
    
    def RSS_to_opt(params, ar_ord, ma_ord, data, exact):
        if exact:
            W_arma = params[:ar_ord+ma_ord+1]
            start_shocks = params[ar_ord+ma_ord+1:]
        else:
            W_arma = params
            start_shocks = np.zeros((ma_ord))
        if exact: return ARMA.RSS(W_arma, start_shocks, ar_ord, ma_ord, data)[0]
        return ARMA.RSS(W_arma, start_shocks, ar_ord, ma_ord, data)
    
    def optres_unpack(params, ar_ord, ma_ord, exact):
        if exact:
            W_arma = params[:ar_ord+ma_ord+1]
            start_shocks = params[ar_ord+ma_ord+1:]
        else:
            W_arma = params
            start_shocks = np.zeros((ma_ord))
        return W_arma, start_shocks
    
    def fit(self, rand_steps=3, solver="l-bfgs-b", maxiter=500, maxfun=15000, tol=1e-4, 
            iprint=0, exact=True, jac=True, rand_init=True):
        tik = time()
        #--optimization---
        rss_min, count, best_optres = np.inf, 0, None
        while count < rand_steps:
            try:
                if exact:
                    if rand_init: init = np.random.normal(size=self.num_of_params+self.ma_ord)
                    else: init = np.zeros((self.num_of_params+self.ma_ord))
                    optres = optimize.minimize(ARMA.RSS_to_opt, init, 
                                               (self.ar_ord, self.ma_ord, self.train, exact), 
                                               method=solver, jac=jac, 
                                               options={"gtol":tol, "maxfun":maxfun, 
                                                        "maxiter":maxiter, "iprint":iprint})
                else:
                    if rand_init: init = np.random.normal(size=self.num_of_params)
                    else: init = np.zeros((self.num_of_params))
                    optres = optimize.minimize(ARMA.RSS_to_opt, init, 
                                               (self.ar_ord, self.ma_ord, self.train, exact), 
                                               method=solver, jac=jac, 
                                               options={"gtol":tol, "maxfun":maxfun, 
                                                        "maxiter":maxiter, "iprint":iprint})
            except FloatingPointError:
                print("Error on init:{}".format(init))
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
        self.W_arma, self.start_shocks = ARMA.optres_unpack(best_optres.x, self.ar_ord, self.ma_ord, exact)
        self.pred, self.shocks = ARMA.RSS(self.W_arma, self.start_shocks, self.ar_ord, self.ma_ord, self.train, ret_pred_shocks=True)
        self.std_dev = np.sqrt(np.sum(np.square(self.shocks-np.mean(self.shocks)))/(self.shocks.shape[0]-1))
        self.loglik = -0.5*(self.train.shape[0]-self.ar_ord)*np.log(2*np.pi*self.std_dev**2)-best_optres.fun/self.std_dev**2
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
            arma_in = np.hstack((1, data[i-1:i-self.ar_ord-1:-1], shocks[i-1:i-self.ma_ord-1:-1]))
            predicted[i-m] = np.dot(arma_in, self.W_arma)
            shocks[i] = test[i-m]-predicted[i-m]
        self.test_mse = np.sum(np.square(shocks[m:]))/data.shape[0]
        self.test_rmse = np.sqrt(self.test_mse)
        return predicted
    
    def forecast(self, n_pred): pass