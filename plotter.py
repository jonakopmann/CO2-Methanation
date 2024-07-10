import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class Plotter:
    def plot(self, X, Y, data, xlabel, ylabel, zlabel, title):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, data, cmap=cm.coolwarm)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_zlabel(zlabel)

        fig.show()
