import os.path

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
from cmcrameri import cm
from matplotlib import animation
from matplotlib.ticker import AutoMinorLocator

from parameters import Parameters
from thermo import *


class Plotter:
    def __init__(self, t, r, res_final, params: Parameters):
        self.params = params
        self.t = t
        self.r = r
        r_steps = len(r) - 1
        res_x = ca.vertsplit(res_final['xf'], r_steps)
        res_z = ca.vertsplit(res_final['zf'])

        self.w_co2 = ca.vertcat(res_x[0], res_z[0]).full()
        self.w_h2 = ca.vertcat(res_x[2], res_z[2]).full()
        self.w_ch4 = ca.vertcat(res_x[1], res_z[1]).full()
        self.w_h2o = 1 - self.w_co2 - self.w_h2 - self.w_ch4
        self.T = ca.vertcat(res_x[3], res_z[3]).full()
        self.p = ca.vertcat(ca.vertcat(*res_z[-r_steps:]), res_z[-r_steps:][-1]).full()

        self.w_co2_fl = res_z[4].full()
        self.w_h2_fl = res_z[6].full()
        self.w_ch4_fl = res_z[5].full()
        self.w_h2o_fl = 1 - self.w_co2_fl - self.w_h2_fl - self.w_ch4_fl
        self.T_fl = res_z[7].full()
        self.D_co2_eff = res_z[8:8+r_steps]

        self.set_params()

        # Colors
        self.colors = plt.cm.Dark2(np.linspace(0, 1, 8))
        self.cmap = cm.bamako

        M = (self.w_co2 / self.params.M_co2 + self.w_h2 / self.params.M_h2
             + self.w_ch4 / self.params.M_ch4 + self.w_h2o / self.params.M_h2o) ** -1

        self.y_co2 = w_to_y(self.w_co2, self.params.M_co2, M)
        self.y_h2 = w_to_y(self.w_h2, self.params.M_h2, M)
        self.y_ch4 = w_to_y(self.w_ch4, self.params.M_ch4, M)
        self.y_h2o = w_to_y(self.w_h2o, self.params.M_h2o, M)

        ny_fl = (self.w_co2_fl * get_ny_co2(self.T_fl, self.p[-1]) + self.w_h2_fl * get_ny_h2(self.T_fl, self.p[-1])
                 + self.w_ch4_fl * get_ny_ch4(self.T_fl, self.p[-1]) + self.w_h2o_fl * get_ny_h2o(self.T_fl, self.p[-1]))  # [mm^2/s]
        Re = self.params.v * self.params.r_max * 2 / ny_fl
        Sc_co2 = ny_fl / self.D_co2_eff[-1]
        Sh_co2 = 2 + 0.6 * Re ** 0.5 + Sc_co2 ** (1 / 3)
        self.beta_co2 = (Sh_co2 * self.D_co2_eff[-1] / (2 * self.params.r_max)).full()  # [mm/s]

        M_fl = (self.w_co2_fl / self.params.M_co2 + self.w_h2_fl / self.params.M_h2
                + self.w_ch4_fl / self.params.M_ch4 + self.w_h2o_fl / self.params.M_h2o) ** -1
        roh_fl = self.p[-1] * 1e5 * M_fl / (self.params.R * self.T_fl)
        roh = self.p[-1] * 1e5 * M[-1, :] / (self.params.R * self.T[-1, :])

        A = 5  # [mm^2]
        self.n_in = self.params.v * self.w_co2_fl * 1e-9 * roh_fl / M_fl * A

        self.n_aus = self.beta_co2 * (self.w_co2[-1, :] * roh - self.w_co2_fl * roh_fl) * 1e-9 / self.params.M_co2 * 4 * np.pi * self.params.r_max ** 2

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
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

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
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        def anim(t):
            line_co2.set_ydata(self.w_co2[:, t])
            line_h2.set_ydata(self.w_h2[:, t])
            line_ch4.set_ydata(self.w_ch4[:, t])
            line_h2o.set_ydata(self.w_h2o[:, t])
            time_text.set_text('t = {:.2f}'.format(self.t[t]))  # update the time text

        ani = animation.FuncAnimation(fig, func=anim, frames=len(self.t),
                                      interval=max(self.t) / len(self.t) * 1000 * scale)

        dir_name = os.path.dirname(file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

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
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

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

        ax.plot(self.r, self.y_co2, label=r'$y_\mathrm{CO_2}$')
        ax.plot(self.r, self.y_h2, label=r'$y_\mathrm{H_2}$')
        ax.plot(self.r, self.y_ch4, label=r'$y_\mathrm{CH_4}$')
        ax.plot(self.r, self.y_h2o, label=r'$y_\mathrm{H_2O}$')

        # set title
        ax.set_title(title)
        ax.set_xlabel('r / mm')
        ax.set_ylabel(r'$y_\mathrm{i}$')
        ax.legend()
        ax.grid()
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        fig.show()

    def animate_y(self, file, title, scale=1.0):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()

        # set limits for x and y axis (r and w)
        ax.set_xlim(min(self.r), max(self.r))
        ax.set_ylim(0, 1)

        line_co2, = ax.plot(self.r, self.y_co2[:, 0], label=r'$y_\mathrm{CO_2}$')
        line_h2, = ax.plot(self.r, self.y_h2[:, 0], label=r'$y_\mathrm{H_2}$')
        line_ch4, = ax.plot(self.r, self.y_ch4[:, 0], label=r'$y_\mathrm{CH_4}$')
        line_h2o, = ax.plot(self.r, self.y_h2o[:, 0], label=r'$y_\mathrm{H_2O}$')

        # add a text element to display the time
        time_text = ax.text(0.2, 0.9, 't = {:.2f}'.format(self.t[0]), transform=ax.transAxes)

        # set title
        ax.set_title(title)
        ax.set_xlabel('r / mm')
        ax.set_ylabel(r'$y_\mathrm{i}$')
        ax.legend(loc='upper left')
        ax.grid()
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        def anim(t):
            line_co2.set_ydata(self.y_co2[:, t])
            line_h2.set_ydata(self.y_h2[:, t])
            line_ch4.set_ydata(self.y_ch4[:, t])
            line_h2o.set_ydata(self.y_h2o[:, t])
            time_text.set_text('t = {:.2f}'.format(self.t[t]))  # update the time text

        ani = animation.FuncAnimation(fig, func=anim, frames=len(self.t),
                                      interval=max(self.t) / len(self.t) * 1000 * scale)

        dir_name = os.path.dirname(file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        ani.save(file)

    def animate_Y(self, file, title, scale=1.0):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()

        # set limits for x and y axis (r and w)
        ax.set_xlim(min(self.r), max(self.r))
        ax.set_ylim(0, 1)

        line, = ax.plot(self.r, (self.y_ch4[:, 0] - self.y_ch4[:, 0]) / self.y_co2[:, 0] * self.v_co2 / self.v_ch4)

        # add a text element to display the time
        time_text = ax.text(0.2, 0.9, 't = {:.2f}'.format(self.t[0]), transform=ax.transAxes)

        # set title
        ax.set_title(title)
        ax.set_xlabel('r / mm')
        ax.set_ylabel(r'$Y_{\mathrm{CH}_4,\mathrm{CO}_2}$')
        ax.grid()
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        def anim(t):
            line.set_ydata((self.y_ch4[:, 0] - self.y_ch4[:, t]) / self.y_co2[:, 0] * self.v_co2 / self.v_ch4)
            time_text.set_text('t = {:.2f}'.format(self.t[t]))  # update the time text

        ani = animation.FuncAnimation(fig, func=anim, frames=len(self.t),
                                      interval=max(self.t) / len(self.t) * 1000 * scale)

        dir_name = os.path.dirname(file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        ani.save(file)

    def animate_X_co2(self, title):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.plot(self.t, ((self.n_in - self.n_aus) / self.n_in).flatten())

        # set title
        ax.set_title(title)
        ax.set_xlabel('t / s')
        ax.set_ylabel(r'$X_{\mathrm{CO_2}}$')
        ax.grid()
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        fig.show()

    def plot_T_fl_surf(self):
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(self.t, self.T_fl.flatten())
        axs[0].set_ylabel('T / K')

        axs[1].plot(self.t, self.T[0, :].flatten())
        axs[1].set_ylabel('T / K')
        axs[1].set_xlabel('t / s')

        fig.show()

    def plot_w_co2_fl_surf(self):
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(self.t, self.w_co2_fl.flatten())
        axs[0].set_ylabel(r'$w_{\mathrm{CO_2}}$')

        axs[1].plot(self.t, self.w_co2[0, :].flatten())
        axs[1].set_ylabel(r'$w_{\mathrm{CO_2}}$')
        axs[1].set_xlabel('t / s')

        fig.show()

    def plot_w_h2_fl_surf(self):
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(self.t, self.w_h2_fl.flatten())
        axs[0].set_ylabel(r'$w_{\mathrm{H_2}}$')

        axs[1].plot(self.t, self.w_h2[0, :].flatten())
        axs[1].set_ylabel(r'$w_{\mathrm{H_2}}$')
        axs[1].set_xlabel('Time / s')

        fig.show()
