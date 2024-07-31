import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib import animation

from thermo import w_to_y


class Plotter:
    def __init__(self, t, r, w_co2, w_h2, w_ch4, w_h2o, T, p):
        self.t = t
        self.r = r
        self.w_co2 = w_co2
        self.w_h2 = w_h2
        self.w_ch4 = w_ch4
        self.w_h2o = w_h2o
        self.T = T
        self.p = p
        self.set_params()

    def set_params(self):
        # plt.rcParams['figure.figsize'] = (fig_w, fig_h)
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

    def plot_w(self, t, title):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()

        # set limits for x and y axis (r and w)
        ax.set_xlim(min(self.r), max(self.r))
        ax.set_ylim(0, 1)

        ax.plot(self.r, self.w_co2[:, t], label=r'$w_\mathrm{CO_2}$')
        ax.plot(self.r, self.w_h2[:, t], label=r'$w_\mathrm{H_2}$')
        ax.plot(self.r, self.w_ch4[:, t], label=r'$w_\mathrm{CH_4}$')
        ax.plot(self.r, self.w_h2o[:, t], label=r'$w_\mathrm{H_2O}$')

        # set title
        ax.set_title(title)
        ax.set_xlabel('r / mm')
        ax.set_ylabel(r'$w_\mathrm{i}$')
        ax.legend()
        ax.grid()

        fig.show()

    def animate_w(self, file, title, scale=1.0):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()

        # set limits for x and y axis (r and w)
        ax.set_xlim(min(self.r), max(self.r))
        ax.set_ylim(0, 1)

        line_co2, = ax.plot(self.r, self.w_co2[:, 0], label=r'$w_\mathrm{CO_2}$')
        line_h2, = ax.plot(self.r, self.w_h2[:, 0], label=r'$w_\mathrm{H_2}$')
        line_ch4, = ax.plot(self.r, self.w_ch4[:, 0], label=r'$w_\mathrm{CH_4}$')
        line_h2o, = ax.plot(self.r, self.w_h2o[:, 0], label=r'$w_\mathrm{H_2O}$')

        # add a text element to display the time
        time_text = ax.text(0.2, 0.9, 't = {:.2f}'.format(self.t[0]), transform=ax.transAxes)

        # set title
        ax.set_title(title)
        ax.set_xlabel('r / mm')
        ax.set_ylabel(r'$w_\mathrm{i}$')
        ax.legend(loc='upper left')
        ax.grid()

        def anim(t):
            line_co2.set_ydata(self.w_co2[:, t])
            line_h2.set_ydata(self.w_h2[:, t])
            line_ch4.set_ydata(self.w_ch4[:, t])
            line_h2o.set_ydata(self.w_h2o[:, t])
            time_text.set_text('t = {:.2f}'.format(self.t[t]))  # update the time text

        ani = animation.FuncAnimation(fig, func=anim, frames=len(self.t),
                                      interval=max(self.t) / len(self.t) * 1000 * scale)
        ani.save(file)

    def animate_T(self, file, title, scale=1.0):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()

        # set limits for x and y axis (r and w)
        ax.set_xlim(min(self.r), max(self.r))
        min_T = np.min(self.T)
        max_T = np.max(self.T)
        ax.set_ylim(min_T - 0.05 * (max_T - min_T), max_T + 0.05 * (max_T - min_T))

        line, = ax.plot(self.r, self.T[:, 0])

        # add a text element to display the time
        time_text = ax.text(0.05, 0.9, 't = {:.2f}'.format(self.t[0]), transform=ax.transAxes)

        # set title
        ax.set_title(title)
        ax.set_xlabel('r / mm')
        ax.set_ylabel('T / K')
        ax.grid()

        # ax.legend(loc='upper left')

        def anim(t):
            line.set_ydata(self.T[:, t])
            time_text.set_text('t = {:.2f}'.format(self.t[t]))  # update the time text

        ani = animation.FuncAnimation(fig, func=anim, frames=len(self.t),
                                      interval=max(self.t) / len(self.t) * 1000 * scale)
        ani.save(file)

    def plot_3d(self, Z, label, title, zmin=None, zmax=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.set_xlim(min(self.t), max(self.t))
        ax.set_ylim(min(self.r), max(self.r))
        if zmin is not None and zmax is not None:
            ax.set_zlim(zmin, zmax)

        # create meshgrid and plot
        X, Y = np.meshgrid(self.t, self.r)
        ax.plot_surface(X, Y, Z, cmap=self.cmap)

        # set title and labels
        ax.set_title(title)
        ax.set_xlabel('t / s')
        ax.set_ylabel('r / mm')
        ax.set_zlabel(label)

        # show
        fig.show()

    def plot_3d_all(self):
        self.plot_3d(self.w_co2, r'$w_\mathrm{CO_2}$', r'$\mathrm{CO_2}$', 0, 1)
        self.plot_3d(self.w_h2, r'$w_\mathrm{H_2}$', r'$\mathrm{H_2}$', 0, 1)
        self.plot_3d(self.w_ch4, r'$w_\mathrm{CH_4}$', r'$\mathrm{CH_4}$', 0, 1)
        self.plot_3d(self.w_h2o, r'$w_\mathrm{H_2O}$', r'$\mathrm{H_2O}$', 0, 1)
        self.plot_3d(self.T, 'T / K', 'Temperature')
        self.plot_3d(self.p, 'p / bar', 'Pressure')

    def plot_hm(self, Z, label, title, zmin, zmax):
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set_xlim(min(self.t), max(self.t))
        ax.set_ylim(min(self.r), max(self.r))

        # create meshgrid and plot
        X, Y = np.meshgrid(self.t, self.r)
        hm = ax.pcolor(X, Y, Z, cmap=self.cmap, vmin=zmin, vmax=zmax)

        # set title and labels
        ax.set_title(title)
        ax.set_xlabel('t / s')
        ax.set_ylabel('r / mm')

        cb = fig.colorbar(hm, orientation='horizontal')
        cb.set_label(label)

        # show
        fig.show()

    def plot_hm_all(self):
        self.plot_hm(self.w_co2, r'$w_\mathrm{CO_2}$', r'$\mathrm{CO_2}$', 0, 1)
        self.plot_hm(self.w_h2, r'$w_\mathrm{H_2}$', r'$\mathrm{H_2}$', 0, 1)
        self.plot_hm(self.w_ch4, r'$w_\mathrm{CH_4}$', r'$\mathrm{CH_4}$', 0, 1)
        self.plot_hm(self.w_h2o, r'$w_\mathrm{H_2O}$', r'$\mathrm{H_2O}$', 0, 1)

    def plot_y(self, title, t):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()

        # set limits for x and y axis (r and w)
        ax.set_xlim(min(self.r), max(self.r))
        ax.set_ylim(0, 1)

        M_co2 = 44.0095  # [g/mol]
        M_h2 = 2.01588  # [g/mol]
        M_ch4 = 16.0425  # [g/mol]
        M_h2o = 18.0153  # [g/mol]

        M = (self.w_co2[:, t] / M_co2 + self.w_h2[:, t] / M_h2
             + self.w_ch4[:, t] / M_ch4 + self.w_h2o[:, t] / M_h2o) ** -1

        ax.plot(self.r, w_to_y(self.w_co2[:, t], M_co2, M), label=r'$y_\mathrm{CO_2}$')
        ax.plot(self.r, w_to_y(self.w_h2[:, t], M_h2, M), label=r'$y_\mathrm{H_2}$')
        ax.plot(self.r, w_to_y(self.w_ch4[:, t], M_ch4, M), label=r'$y_\mathrm{CH_4}$')
        ax.plot(self.r, w_to_y(self.w_h2o[:, t], M_h2o, M), label=r'$y_\mathrm{H_2O}$')

        # set title
        ax.set_title(title)
        ax.set_xlabel('r / mm')
        ax.set_ylabel(r'$y_\mathrm{i}$')
        ax.legend()
        ax.grid()

        fig.show()