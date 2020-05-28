import arma
import nn
import armann

a = arma.ARMA([], (1, 1), 0)
b = nn.NN([], (1, 2, 3), 0)
c = armann.ARMA_NN([], (1, 1, 1, 2, 3), 0)
print(a)
print(b)
print(c)