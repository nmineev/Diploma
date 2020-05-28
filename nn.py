import numpy as np
import scipy.optimize as optimize
from time import time

class NN:
    def __init__(self, data, order, test_size):
        self.order = order
        self.ar_ord, self.ma_ord, self.hid_size = order
        self.num_of_params = 1+(self.ar_ord+self.ma_ord+2)*self.hid_size
        data = np.array(data)
        self.test_index = int(data.shape[0]*(1-test_size))
        self.train, self.test = data[:self.test_index], data[self.test_index:]
    
    @staticmethod
    def F(vect, add=True):
        logistic = lambda z: 1/(1+np.exp(-z))
        for i in range(vect.shape[0]):
            try:
                vect[i] = logistic(vect[i])
            except FloatingPointError:
                if vect[i]>0: vect[i] = 1
                else: vect[i] = 0
        if add: return np.hstack((1, vect))
        return vect
    
    def derF(vect): return NN.F(vect, add=False)-NN.to_square(NN.F(vect, add=False))
    
    def to_square(vect):
        for i in range(vect.shape[0]):
            try:
                vect[i] = vect[i]**2
            except FloatingPointError:
                vect[i] = np.inf
        return vect
    
    @staticmethod
    def RSS(W1, W2, start_shocks, ar_ord, ma_ord, hid_size, data, ret_pred_shocks=False):
        shocks = np.hstack((0, start_shocks))
        rss = 0
        delta1, delta2 = np.zeros((hid_size)), 0
        grad_W1 = np.zeros((ar_ord+ma_ord+1, hid_size))
        grad_W2 = np.zeros((hid_size+1))
        if ret_pred_shocks:
            first_shocks = np.append(np.zeros((ar_ord-ma_ord if ar_ord-ma_ord > 0 else 0)), start_shocks[ma_ord-ar_ord if ma_ord-ar_ord>0 else 0:])
            pred_to_ret = np.append(data[:ar_ord]-first_shocks, np.zeros((data.shape[0]-ar_ord)))
            shocks_to_ret = np.append(first_shocks, np.zeros((data.shape[0]-ar_ord)))
        #-----loop-start---
        I = np.hstack((1, data[ar_ord-1:None if ar_ord else -1:-1], shocks[-1:-ma_ord-1:-1]))
        A1 = np.dot(I, W1)
        Z2 = NN.F(A1)
        pred = np.dot(Z2, W2)
        shocks = np.append(shocks[1:], data[ar_ord]-pred)
        delta2 = -shocks[-1]
        delta1 = NN.derF(A1)*W2[1:]*delta2
        grad_W1 += np.tensordot(I, delta1, axes=0)
        grad_W2 += Z2*delta2
        if ret_pred_shocks:
            pred_to_ret[ar_ord] = pred
            shocks_to_ret[ar_ord] = shocks[-1]
        try:
            rss += shocks[-1]**2
        except FloatingPointError:
            rss = np.inf
        for i in range(ar_ord+1, data.shape[0]):
            I = np.hstack((1, data[i-1:i-ar_ord-1:-1], shocks[-1:-ma_ord-1:-1]))
            A1 = np.dot(I, W1)
            Z2 = NN.F(A1)
            pred = np.dot(Z2, W2)
            shocks = np.append(shocks[1:], data[i]-pred)
            delta2 = -shocks[-1]
            delta1 = NN.derF(A1)*W2[1:]*delta2
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
        return 0.5*(rss), np.hstack((grad_W1.ravel(), grad_W2))
    
    def RSS_to_opt(params, ar_ord, ma_ord, hid_size, data, exact):
        W1 = params[:(ar_ord+ma_ord+1)*hid_size].reshape((ar_ord+ma_ord+1, hid_size))
        if exact:
            W2 = params[(ar_ord+ma_ord+1)*hid_size:(ar_ord+ma_ord+1)*hid_size+hid_size+1]
            start_shocks = params[(ar_ord+ma_ord+1)*hid_size+hid_size+1:]
        else:
            W2 = params[-hid_size-1:]
            start_shocks = np.zeros((ma_ord))
        if exact: return NN.RSS(W1, W2, start_shocks, ar_ord, ma_ord, hid_size, data)[0]
        return NN.RSS(W1, W2, start_shocks, ar_ord, ma_ord, hid_size, data)
    
    def optres_unpack(params, ar_ord, ma_ord, hid_size, exact):
        W1 = params[:(ar_ord+ma_ord+1)*hid_size].reshape((ar_ord+ma_ord+1, hid_size))
        if exact:
            W2 = params[(ar_ord+ma_ord+1)*hid_size:(ar_ord+ma_ord+1)*hid_size+hid_size+1]
            start_shocks = params[(ar_ord+ma_ord+1)*hid_size+hid_size+1:]
        else:
            W2 = params[-hid_size-1:]
            start_shocks = np.zeros((ma_ord))
        return W1, W2, start_shocks   
    
    def fit(self, rand_steps=3, solver="l-bfgs-b", maxiter=500, maxfun=15000, tol=1e-4, iprint=0, exact=True, jac=True, rand_init=True):
        tik = time()
        #--optimization---
        rss_min, count, best_optres = np.inf, 0, None
        while count < rand_steps:
            try:
                if exact:
                    if rand_init: init = np.random.normal(size=self.num_of_params+self.ma_ord)
                    else: init = np.zeros((self.num_of_params+self.ma_ord))
                    optres = optimize.minimize(NN.RSS_to_opt, init, (self.ar_ord, self.ma_ord, self.hid_size, self.train, exact), method=solver, jac=jac, options={"gtol":tol, "maxfun":maxfun, "maxiter":maxiter, "iprint":iprint})
                else:
                    if rand_init: init = np.random.normal(size=self.num_of_params)
                    else: init = np.zeros((self.num_of_params))
                    optres = optimize.minimize(NN.RSS_to_opt, init, (self.ar_ord, self.ma_ord, self.hid_size, self.train, exact), method=solver, jac=jac, options={"gtol":tol, "maxfun":maxfun, "maxiter":maxiter, "iprint":iprint})
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
        self.W1, self.W2, self.start_shocks = NN.optres_unpack(best_optres.x, self.ar_ord, self.ma_ord, self.hid_size, exact)
        self.pred, self.shocks = NN.RSS(self.W1, self.W2, self.start_shocks, self.ar_ord, self.ma_ord, self.hid_size, self.train, ret_pred_shocks=True)
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
            I = np.hstack((1, data[i-1:i-self.ar_ord-1:-1], shocks[i-1:i-self.ma_ord-1:-1]))
            predicted[i-m] = np.dot(NN.F(np.dot(I, self.W1)), self.W2)
            shocks[i] = test[i-m]-predicted[i-m]
        self.test_mse = np.sum(np.square(shocks[m:]))/data.shape[0]
        self.test_rmse = np.sqrt(self.test_mse)
        return predicted
    
    def forecast(self, n_pred): pass