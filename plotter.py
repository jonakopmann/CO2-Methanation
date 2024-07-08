import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class Plotter:
    def __init__(self, fig_width=6, fig_height=4):
        # Font and lable settings
        plt.rcParams['figure.figsize'] = (fig_width, fig_height)
        plt.rcParams['axes.linewidth'] = 1  # set the value globally
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)
        plt.rc('axes', labelsize=11)
        plt.rc('legend', fontsize=9)
        plt.rcParams['lines.markersize'] = 5
        plt.rcParams['lines.linewidth'] = 1
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

    def plot(xlabel, ylabel):
        fig, ax = plt.subplots(1, 1, sharex=True, constrained_layout=False)

        # Legend
        ax.legend(frameon=True, fancybox=False, loc='best', ncol=2, framealpha=1, edgecolor='black')
        # Axis
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
