import numpy as np
import matplotlib.pyplot as plt
import unittest

class TestPerceptron(unittest.TestCase):
    def testPlotRandomOR(self):
        np.random.seed( 0)
        X_xor = np.random.randn( 200, 2)
        y_xor = np.logical_xor( X_xor[:, 0] > 0, X_xor[:, 1] > 0)
        y_xor = np.where( y_xor, 1, -1)
        plt.scatter( X_xor[y_xor == 1, 0],
            X_xor[ y_xor == 1, 1], c ='b',
            marker ='x',
            label =' 1')
        plt.scatter( X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker= 's',
            label='-1')
        plt.ylim(-3.0)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
