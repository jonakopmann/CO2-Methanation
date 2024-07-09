import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator


class Plotter:
    def plot(self, X, Y, data):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm)
        plt.xlabel('t')
        plt.ylabel('r')
        ax.set_zlabel('y_i')
        ax.set_zlim(max(np.min(data), 0), min(np.max(data), 1))

        fig.show()
