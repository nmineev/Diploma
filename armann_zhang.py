import arma
import nn

from time import time
import numpy as np

class ARMA_NN_Zhang:
    def __init__(self, data, order, test_size):
        self.order = order
        self.ar_ord, self.ma_ord, self.nn_in_size, self.nn_hid_size = order
        self.num_of_params = 2+self.ar_ord+self.ma_ord+(self.nn_in_size+2)*self.nn_hid_size
        data = np.array(data)
        self.test_index = int(data.shape[0]*(1-test_size))
        self.train, self.test = data[:self.test_index], data[self.test_index:]
    
    def fit(self, rand_steps=3, solver="l-bfgs-b", maxiter=500, maxfun=15000, tol=1e-4, iprint=0, exact=True, jac=True, rand_init=True):
        tik = time()
        #--optimization---
        self.arma_comp = arma.ARMA(self.train, (self.ar_ord, self.ma_ord), 0).fit(rand_steps=rand_steps, solver=solver, maxiter=maxiter, maxfun=maxfun, tol=tol, iprint=iprint, exact=exact, jac=jac, rand_init=rand_init)
        self.nn_comp = nn.NN(self.arma_comp.shocks, (self.nn_in_size, 0, self.nn_hid_size), 0).fit(rand_steps=rand_steps, solver=solver, maxiter=maxiter, maxfun=maxfun, tol=tol, iprint=iprint, exact=exact, jac=jac, rand_init=rand_init)
        #------------
        print("fit-time:{}".format(time()-tik))
        self.W_arma, self.W1, self.W2, self.start_shocks = self.arma_comp.W_arma, self.nn_comp.W1, self.nn_comp.W2, self.arma_comp.start_shocks
        self.pred, self.shocks = self.arma_comp.pred+self.nn_comp.pred, self.nn_comp.shocks
        self.std_dev = np.sqrt(np.sum(np.square(self.shocks-np.mean(self.shocks)))/(self.shocks.shape[0]-1))
        self.loglik = -0.5*(self.train.shape[0]-self.ar_ord)*np.log(2*np.pi*self.std_dev**2)-0.5*np.sum(np.square(self.shocks[self.ar_ord:]))/self.std_dev**2
        self.aic = -2/self.train.shape[0]*(self.loglik-self.num_of_params)
        self.mse = np.sum(np.square(self.shocks[self.ar_ord:]))/self.train.shape[0]
        self.rmse = np.sqrt(self.mse)
        return self
        
    def predict(self, test=None):
        if test is None: test = self.test
        arma_pred = self.arma_comp.predict(test)
        arma_shocks = test-arma_pred
        nn_pred = self.nn_comp.predict(arma_shocks)
        predicted = arma_pred+nn_pred
        shocks = test-predicted
        self.test_mse = np.sum(np.square(shocks))/test.shape[0]
        self.test_rmse = np.sqrt(self.test_mse)
        return predicted
    
    def forecast(self, n_pred): pass