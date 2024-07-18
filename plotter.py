import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm


class Plotter:
    def __init__(self, fig_w=7, fig_h=3.5, fig_pad=0.1):
        #plt.rcParams['figure.figsize'] = (fig_w, fig_h)
        plt.rcParams['axes.linewidth'] = 1  # set the value globally
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('axes', labelsize=11)
        plt.rc('legend', fontsize=11)
        plt.rcParams['lines.markersize'] = 5
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 1
        plt.rcParams['ytick.minor.size'] = 3
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['legend.edgecolor'] = 'k'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams["legend.framealpha"] = 1
        plt.rcParams['xtick.major.pad'] = 8
        plt.rcParams['ytick.major.pad'] = 8
        plt.rcParams['legend.handletextpad'] = 0.4
        plt.rcParams['legend.columnspacing'] = 0.5
        plt.rcParams['legend.labelspacing'] = 0.3
        plt.rcParams['legend.title_fontsize'] = 14
        # Colors
        self.colors = plt.cm.Dark2(np.linspace(0, 1, 8))
        self.cmap = cm.bamako

    def plot(self, X, Y, data, xlabel, ylabel, zlabel, title, zmin=None, zmax=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # set axis limits
        ax.set_xlim(min(X), max(X))
        ax.set_ylim(min(Y), max(Y))
        if zmin is not None and zmax is not None:
            ax.set_zlim(zmin, zmax)

        # create meshgrid and plot
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, data, cmap=self.cmap, linewidth=0)

        # set title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        # show
        fig.show()
