import os.path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cmcrameri import cm
from matplotlib import animation
from matplotlib.ticker import AutoMinorLocator

from diffusion_coefficient import DiffusionCoefficient
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
        self.M = (self.w_co2 / self.params.M_co2 + self.w_h2 / self.params.M_h2
             + self.w_ch4 / self.params.M_ch4 + self.w_h2o / self.params.M_h2o) ** -1
        self.p = (self.params.M_0 * self.T) / (self.M * self.params.T_0) * self.params.p_0

        self.w_co2_fl = res_z[4].full()
        self.w_h2_fl = res_z[6].full()
        self.w_ch4_fl = res_z[5].full()
        self.w_h2o_fl = 1 - self.w_co2_fl - self.w_h2_fl - self.w_ch4_fl
        self.T_fl = res_z[7].full()

        self.set_params()

        # Colors
        self.colors = plt.cm.Dark2(np.linspace(0, 1, 8))
        # kit colors
        # self.colors = np.array([(167, 130, 46), (163, 16, 124), (35, 161, 224), (140, 182, 60), (252, 229, 0), (223, 155, 27), (162, 34, 35)]) / 255
        self.cmap = cm.bamako
        self.fig_pad = 0.1

        self.rho = self.p * 1e5 * self.M / (self.params.R * self.T)

        self.y_co2 = w_to_y(self.w_co2, self.params.M_co2, self.M)
        self.y_h2 = w_to_y(self.w_h2, self.params.M_h2, self.M)
        self.y_ch4 = w_to_y(self.w_ch4, self.params.M_ch4, self.M)
        self.y_h2o = w_to_y(self.w_h2o, self.params.M_h2o, self.M)


        D_co2_h2 = DiffusionCoefficient.get_D_i_j(self.T, self.p, self.params.M_co2, self.params.M_h2,
                                  self.params.delta_v_co2, self.params.delta_v_h2)
        D_co2_ch4 = DiffusionCoefficient.get_D_i_j(self.T, self.p, self.params.M_co2, self.params.M_ch4,
                                        self.params.delta_v_co2, self.params.delta_v_ch4)
        D_co2_h2o = DiffusionCoefficient.get_D_i_j(self.T, self.p, self.params.M_co2, self.params.M_h2o,
                                        self.params.delta_v_co2, self.params.delta_v_h2o)

        self.D_co2_m = DiffusionCoefficient.get_D_i_m(self.y_co2, self.y_h2, self.y_ch4,
                       self.y_h2o, D_co2_h2, D_co2_ch4, D_co2_h2o)

        M_fl = (self.w_co2_fl / self.params.M_co2 + self.w_h2_fl / self.params.M_h2
                + self.w_ch4_fl / self.params.M_ch4 + self.w_h2o_fl / self.params.M_h2o) ** -1
        p_fl = (self.params.M_0 * self.T_fl) / (M_fl * self.params.T_0) * self.params.p_0
        rho_fl = p_fl * 1e5 * M_fl / (self.params.R * self.T_fl)
        rho_surf = self.p[-1] * 1e5 * self.M[-1, :] / (self.params.R * self.T[-1, :])

        nu_fl = (1e9 / rho_fl
                 * (self.w_co2_fl * get_eta_co2(self.T_fl) + self.w_h2_fl * get_eta_h2(self.T_fl)
                    + self.w_ch4_fl * get_eta_ch4(self.T_fl) + self.w_h2o_fl * get_eta_h2o(
                            self.T_fl)))  # [mm^2/s]
        Re = self.params.v * self.params.r_max * 2 / nu_fl
        Sc_co2 = nu_fl / self.D_co2_m[-1]
        Sh_co2 = 2 + 0.6 * Re ** 0.5 + Sc_co2 ** (1 / 3)
        self.beta_co2 = Sh_co2 * self.D_co2_m[-1] / (2 * self.params.r_max)  # [mm/s]

        self.y_co2_fl = w_to_y(self.w_co2_fl, self.params.M_co2, M_fl)
        self.y_h2_fl = w_to_y(self.w_h2_fl, self.params.M_h2, M_fl)
        self.y_ch4_fl = w_to_y(self.w_ch4_fl, self.params.M_ch4, M_fl)
        self.y_h2o_fl = w_to_y(self.w_h2o_fl, self.params.M_h2o, M_fl)

        n = self.beta_co2 * (self.y_co2_fl - self.y_co2[-1, :]) * 4 * np.pi * (self.params.r_max ** 2)
        n2 = self.params.v * np.pi * (self.params.r_max * self.params.factor) ** 2 * self.y_co2_fl

        a = n / n2

        self.X_co2 = (self.beta_co2 * (self.w_co2_fl * rho_fl - self.w_co2[-1, :] * rho_surf) * 4 / (self.params.v * self.w_co2_fl * rho_fl * self.params.factor ** 2)).flatten()

    def set_params(self):
        plt.rcParams['figure.figsize'] = (3.5, 3.5)
        plt.rcParams['axes.linewidth'] = 1  # set the value globally
        mpl.use('pgf')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['text.usetex'] = True
        plt.rcParams['pgf.rcfonts'] = False
        plt.rcParams['pgf.texsystem'] = 'pdflatex'
        plt.rcParams['pgf.preamble'] = r'\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage[scaled]{helvet}\usepackage{mathptmx}\usepackage[version=3,arrows=pgf-filled]{mhchem}\usepackage{siunitx}\sisetup{detect-all=true}\usepackage{textcomp}\sisetup{per-mode=reciprocal,output-decimal-marker = {.},exponent-product = \cdot,list-units = single,range-units = single,sticky-per = true}'
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

    def plot_w(self, t):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()

        # set limits for x and y axis (r and w)
        ax.set_xlim(min(self.r), max(self.r))
        #ax.set_ylim(0, 1)

        ax.plot(self.r, self.w_co2[:, t], label=r'$w_{\ce{CO2}}$', color=self.colors[0])
        ax.plot(self.r, self.w_h2[:, t], label=r'$w_{\ce{H2}}$', color=self.colors[1])
        ax.plot(self.r, self.w_ch4[:, t], label=r'$w_{\ce{CH4}}$', color=self.colors[2])
        ax.plot(self.r, self.w_h2o[:, t], label=r'$w_{\ce{H2O}}$', color=self.colors[4])

        # set title
        ax.set_xlabel('$r$ / mm')
        ax.set_ylabel(r'$w_i$')
        ax.legend(frameon=True, fancybox=False, loc='best', ncol=1, framealpha=1, edgecolor='black')
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/Closing_H2O_2.pdf', pad_inches=self.fig_pad, bbox_inches='tight')

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

        dir_name = os.path.dirname(file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

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

        line, = ax.plot(self.r, (self.y_ch4[:, 0] - self.y_ch4[:, 0]) / self.y_co2[:, 0] * self.params.v_co2 / self.params.v_ch4)

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
            line.set_ydata((self.y_ch4[:, 0] - self.y_ch4[:, t]) / self.y_co2[:, 0] * self.params.v_co2 / self.params.v_ch4)
            time_text.set_text('t = {:.2f}'.format(self.t[t]))  # update the time text

        ani = animation.FuncAnimation(fig, func=anim, frames=len(self.t),
                                      interval=max(self.t) / len(self.t) * 1000 * scale)

        dir_name = os.path.dirname(file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        ani.save(file)

    def plot_X_co2(self, path):
        # create figure
        fig = plt.figure(hash('X_co2'), constrained_layout=False, figsize=(6, 3))
        index = len(fig.axes) + 1
        if index < 3:
            ax = fig.add_subplot(1, 2, index, sharey=fig.axes[0] if index == 2 else None)
            #ax.set_ylim(0, 0.08)
            ax.set_xlim(min(self.t), max(self.t))

            ax.set_xlabel(r'$t / \si{\s}$')
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        else:
            ax = fig.axes[0 if path == 'sin' else 1]

        #self.X_co2[(self.X_co2 < 0) | (self.X_co2 > 0.1)] = None
        ax.plot(self.t, self.X_co2, color=self.colors[0 if index < 3 else 1])

        if index == 2:
            ax.text(-0.15, 1.06, '(b)', transform=ax.transAxes, fontsize=12, weight='bold')
            ax.label_outer()
        elif index == 1:
            ax.text(-0.25, 1.06, '(a)', transform=ax.transAxes, fontsize=12, weight='bold')
            ax.set_ylabel(r'$X_{\ce{CO_2}}$')
            fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/Conversion.pdf',
                        pad_inches=self.fig_pad, bbox_inches='tight')
        # elif path == 'step':
            # fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/Conversion.pdf',
            #             pad_inches=self.fig_pad, bbox_inches='tight')

    def get_k(self, T):
        # k_0_ref * exp(E_A/R * (1/T_ref - 1/T))
        return self.params.k_0_ref * ca.exp(self.params.EA / self.params.R * ((1 / self.params.T_ref) - (1 / T)))

    def get_K_oh(self, T):
        # K_x_ref * exp(delta_H_x/R * (1/T_ref - 1/T))
        return (self.params.K_oh_ref
                * ca.exp(self.params.delta_H_oh / self.params.R * ((1 / self.params.T_ref) - (1 / T))))

    def get_K_h2(self, T):
        # K_x_ref * exp(delta_H_x/R * (1/T_ref - 1/T))
        return (self.params.K_h2_ref
                * ca.exp(self.params.delta_H_h2 / self.params.R * ((1 / self.params.T_ref) - (1 / T))))

    def get_K_mix(self, T):
        # K_x_ref * exp(delta_H_x/R * (1/T_ref - 1/T))
        return (self.params.K_mix_ref
                * ca.exp(self.params.delta_H_mix / self.params.R * ((1 / self.params.T_ref) - (1 / T))))

    def get_H_R(self, T):
        H_f_h2o = -241.8264 + get_H_h2o(T)
        H_f_ch4 = -74.87310 + get_H_ch4(T)
        H_f_h2 = 0 + get_H_h2(T)
        H_f_co2 = -393.5224 + get_H_co2(T)

        return 1e3 * (self.params.v_co2 * H_f_co2 + self.params.v_h2 * H_f_h2
                      + self.params.v_ch4 * H_f_ch4 + self.params.v_h2o * H_f_h2o)  # [J/mol]

    def get_K_eq(self, T, p):
        # exp(-delta_R_G/RT)
        S = (self.params.v_co2 * (get_S_co2(T)) + self.params.v_h2 * (get_S_h2(T))
             + self.params.v_ch4 * (get_S_ch4(T)) + self.params.v_h2o * (get_S_h2o(T)))  # [J/(mol*K)]
        G = self.get_H_R(T) - T * S  # [J/mol]
        return ca.exp(-G / (self.params.R * T)) * p ** -2

    def plot_cat_eff(self):
        # create figure
        fig = plt.figure(hash('cat_eff'), constrained_layout=False, figsize=(6, 3))
        index = len(fig.axes) + 1
        ax = fig.add_subplot(1, 2, index, sharey=fig.axes[0] if index == 2 else None)

        ax.set_xlim(min(self.t), max(self.t))

        p_co2 = self.p * self.w_co2 * self.M / self.params.M_co2
        p_h2 = self.p * self.w_h2 * self.M / self.params.M_h2
        p_ch4 = self.p * self.w_ch4 * self.M / self.params.M_ch4
        p_h2o = self.p * self.w_h2o * self.M / self.params.M_h2o

        r = (self.get_k(self.T) * (p_h2 ** 0.5) * (p_co2 ** 0.5) * (
                1 - (p_ch4 * (p_h2o ** 2)) / (p_co2 * (p_h2 ** 4) * self.get_K_eq(self.T, self.p)))
             / ((1 + self.get_K_oh(self.T) * (p_h2o / (p_h2 ** 0.5)) + self.get_K_h2(self.T) *
                 (p_h2 ** 0.5) + self.get_K_mix(self.T) * (p_co2 ** 0.5)) ** 2)).full()

        r = np.nan_to_num(r)

        sum = 0
        for i in range(self.params.r_steps):
            sum += i ** 2 * r[i, :] + (i + 1) ** 2 * r[i + 1, :]
        sum *= (self.params.h ** 3) / 2

        val = (3 / (self.params.r_max ** 3) * sum / r[-1, :]).flatten()

        ax.set_ylim(0.65, 1)

        ax.plot(self.t, val, color=self.colors[0], label=rf'$f_y = \SI{{{self.params.f_y}}}{{\per\s}} \quad f_T = \SI{{{self.params.f_T}}}{{\per\s}}$')

        # set title
        ax.set_xlabel('$t / \si{\s}$')
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        if index == 2:
            ax.text(-0.15, 1.06, '(b)', transform=ax.transAxes, fontsize=12, weight='bold')
            ax.label_outer()
            #fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/CatalystEfficiency.pdf',
                        #pad_inches=self.fig_pad, bbox_inches='tight')
        else:
            ax.text(-0.25, 1.06, '(a)', transform=ax.transAxes, fontsize=12, weight='bold')
            ax.set_ylabel(r'$\eta_{\ce{CH4}}$')

    def plot_cat_eff_2(self, type):
        fig = plt.figure(hash('cat_eff'), constrained_layout=False, figsize=(6, 3))
        if type == 'sin':
            ax = fig.axes[0]
        else:
            ax = fig.axes[1]

        p_co2 = self.p * self.w_co2 * self.M / self.params.M_co2
        p_h2 = self.p * self.w_h2 * self.M / self.params.M_h2
        p_ch4 = self.p * self.w_ch4 * self.M / self.params.M_ch4
        p_h2o = self.p * self.w_h2o * self.M / self.params.M_h2o

        r = (self.get_k(self.T) * (p_h2 ** 0.5) * (p_co2 ** 0.5) * (
                1 - (p_ch4 * (p_h2o ** 2)) / (p_co2 * (p_h2 ** 4) * self.get_K_eq(self.T, self.p)))
             / ((1 + self.get_K_oh(self.T) * (p_h2o / (p_h2 ** 0.5)) + self.get_K_h2(self.T) *
                 (p_h2 ** 0.5) + self.get_K_mix(self.T) * (p_co2 ** 0.5)) ** 2)).full()

        r = np.nan_to_num(r)

        sum = 0
        for i in range(self.params.r_steps):
            sum += i ** 2 * r[i, :] + (i + 1) ** 2 * r[i + 1, :]
        sum *= (self.params.h ** 3) / 2

        val = (3 / (self.params.r_max ** 3) * sum / r[-1, :]).flatten()
        ax.plot(self.t, val, color=self.colors[1], label=rf'$f_y = \SI{{{self.params.f_y}}}{{\per\s}} \quad f_T = \SI{{{self.params.f_T}}}{{\per\s}}$')

        if type == 'step':
            ax.legend(frameon=True, fancybox=False, ncol=1, framealpha=1, edgecolor='black')
            fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/CatalystEfficiency.pdf',
                        pad_inches=self.fig_pad, bbox_inches='tight')

    def plot_cat_eff_thiele(self, title):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()

        p_co2 = self.p[-1, :] * self.w_co2[-1, :] * self.M[-1, :] / self.params.M_co2
        p_h2 = self.p[-1, :] * self.w_h2[-1, :] * self.M[-1, :] / self.params.M_h2
        p_ch4 = self.p[-1, :] * self.w_ch4[-1, :] * self.M[-1, :] / self.params.M_ch4
        p_h2o = self.p[-1, :] * self.w_h2o[-1, :] * self.M[-1, :] / self.params.M_h2o

        r = (self.get_k(self.T[-1, :]) * (p_h2 ** 0.5) * (p_co2 ** 0.5) * (
                1 - (p_ch4 * (p_h2o ** 2)) / (p_co2 * (p_h2 ** 4) * self.get_K_eq(self.T[-1, :], self.p[-1, :])))
             / ((1 + self.get_K_oh(self.T[-1, :]) * (p_h2o / (p_h2 ** 0.5)) + self.get_K_h2(self.T[-1, :]) *
                 (p_h2 ** 0.5) + self.get_K_mix(self.T[-1, :]) * (p_co2 ** 0.5)) ** 2)).full()

        r = np.nan_to_num(r)

        thiele = self.params.r_max * np.sqrt(np.abs(-self.params.v_co2 * r.flatten() * (1 - self.params.epsilon) * self.params.rho_s / (self.D_co2_m[-1].full().flatten() * self.w_co2[-1, :].flatten() * self.rho[-1, :].flatten() / self.params.M_co2)))

        eff = (3 / thiele * (1 / np.tanh(thiele) - 1 / thiele))

        ax.plot(self.t, eff.flatten())

        # set title
        ax.set_title(title)
        ax.set_xlabel('t / s')
        ax.set_ylabel(r'$\eta_{\mathrm{CH_4}}$')
        ax.grid()
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        fig.show()

    def plot_feed_center(self, path):
        fig, (feed_w_ax, feed_T_ax, center_w_ax, center_T_ax) = plt.subplots(4, 1, sharex=True, constrained_layout=False, figsize=(4.5, 6))

        #feed_w_ax.set_ylim(0, 1)
        #center_w_ax.set_ylim(0, 1)

        y_max = max(np.max(self.T_fl), np.max(self.T[0, :]))
        y_min = min(np.min(self.T_fl), np.min(self.T[0, :]))
        y_min = y_min - 0.1 * (y_max - y_min)
        y_max = y_max + 0.1 * (y_max - y_min)
        feed_T_ax.set_ylim(y_min, y_max)
        center_T_ax.set_ylim(y_min, y_max)

        feed_w_ax.set_xlim(np.min(self.t), np.max(self.t))
        center_w_ax.set_xlim(np.min(self.t), np.max(self.t))

        feed_w_ax.plot(self.t, self.w_co2_fl.flatten(), label=r'\ce{CO2}', color=self.colors[1])
        feed_w_ax.plot(self.t, self.w_h2_fl.flatten(), label=r'\ce{H2}', color=self.colors[0])
        feed_w_ax.plot(self.t, self.w_ch4_fl.flatten(), label=r'\ce{CH4}', color=self.colors[2])
        feed_w_ax.plot(self.t, self.w_h2o_fl.flatten(), label=r'\ce{H2O}', color=self.colors[3])
        feed_w_ax.set_ylabel(r'$w_{i, \text{f}}$')
        feed_w_ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        feed_w_ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        center_w_ax.plot(self.t, self.w_co2[0, :].flatten(), label=r'\ce{CO2}', color=self.colors[1])
        center_w_ax.plot(self.t, self.w_h2[0, :].flatten(), label=r'\ce{H2}', color=self.colors[0])
        center_w_ax.plot(self.t, self.w_ch4[0, :].flatten(), label=r'\ce{CH4}', color=self.colors[2])
        center_w_ax.plot(self.t, self.w_h2o[0, :].flatten(), label=r'\ce{H2O}', color=self.colors[3])
        center_w_ax.set_ylabel(r'$w_{i, r = 0}$')
        #center_w_ax.set_xlabel('t / s')
        center_w_ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        center_w_ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        center_w_ax.legend(frameon=True, fancybox=False, ncol=1, framealpha=1, edgecolor='black')

        feed_T_ax.plot(self.t, self.T_fl.flatten())
        feed_T_ax.set_ylabel(r'$T_\text{f} / \si{\K}$')
        feed_T_ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        feed_T_ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        center_T_ax.plot(self.t, self.T[0, :].flatten())
        center_T_ax.set_ylabel(r'$T_{r = 0} / \si{\K}$')
        center_T_ax.set_xlabel(r't / \si{\s}')
        center_T_ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        center_T_ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        fig.savefig(path, pad_inches=self.fig_pad, bbox_inches='tight')

    def plot_feed_center_2(self):
        fig = plt.figure(hash('feed_center'), figsize=(6, 6))

        index = 0 if len(fig.axes) == 0 else 1
        center_w_ax = fig.add_subplot(4, 2, index + 1, sharey=fig.axes[0] if index == 1 else None)
        center_T_ax = fig.add_subplot(4, 2, index + 3, sharex=center_w_ax, sharey=fig.axes[1] if index == 1 else None)
        feed_w_ax = fig.add_subplot(4, 2, index + 5, sharex=center_w_ax, sharey=fig.axes[2] if index == 1 else None)
        feed_T_ax = fig.add_subplot(4, 2, index + 7, sharex=center_w_ax, sharey=fig.axes[3] if index == 1 else None)

        feed_w_ax.set_ylim(0, 1)
        center_w_ax.set_ylim(0, 1)

        y_max = max(np.max(self.T_fl), np.max(self.T[0, :]))
        y_min = min(np.min(self.T_fl), np.min(self.T[0, :]))
        y_min = y_min - 0.1 * (y_max - y_min)
        y_max = y_max + 0.1 * (y_max - y_min)
        feed_T_ax.set_ylim(y_min, y_max)
        center_T_ax.set_ylim(y_min, y_max)

        feed_w_ax.set_xlim(np.min(self.t), np.max(self.t))
        center_w_ax.set_xlim(np.min(self.t), np.max(self.t))

        feed_w_ax.plot(self.t, self.y_co2_fl.flatten(), color=self.colors[1])
        feed_w_ax.plot(self.t, self.y_h2_fl.flatten(), color=self.colors[0])

        feed_w_ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        feed_w_ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        center_w_ax.plot(self.t, self.y_co2[0, :].flatten(), color=self.colors[1])
        center_w_ax.plot(self.t, self.y_h2[0, :].flatten(), color=self.colors[0])

        # center_w_ax.set_xlabel('t / s')
        center_w_ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        center_w_ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        feed_T_ax.plot(self.t, self.T_fl.flatten(), color=self.colors[2])
        feed_T_ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        feed_T_ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        center_T_ax.plot(self.t, self.T[0, :].flatten(), color=self.colors[2])

        center_T_ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        center_T_ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        center_w_ax.label_outer()
        center_T_ax.label_outer()
        feed_w_ax.label_outer()
        feed_T_ax.label_outer()
        feed_T_ax.set_xlabel(r't / \si{\s}')

        if index == 1:
            center_w_ax.text(-0.15, 1.06, '(b)', transform=center_w_ax.transAxes, fontsize=12, weight='bold')
            fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/FeedCenter.pdf',
                        pad_inches=self.fig_pad, bbox_inches='tight')
        else:
            center_w_ax.text(-0.25, 1.06, '(a)', transform=center_w_ax.transAxes, fontsize=12, weight='bold')
            center_w_ax.text(self.params.t_max * 0.8, self.y_h2[0, 0], r'\ce{H2}', color=self.colors[0])
            center_w_ax.text(self.params.t_max * 0.8, self.y_co2[0, 0], r'\ce{CO2}', color=self.colors[1])
            center_T_ax.set_ylabel(r'$T_{r = 0} / \si{\K}$')
            feed_T_ax.set_ylabel(r'$T_\text{f} / \si{\K}$')
            center_w_ax.set_ylabel(r'$y_{i, r = 0}$')
            feed_w_ax.set_ylabel(r'$y_{i, \text{f}}$')

    def plot_w_h2_fl_surf(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)

        axs[0].set_xlim(np.min(self.t), np.max(self.t))
        axs[1].set_xlim(np.min(self.t), np.max(self.t))

        axs[0].plot(self.t, self.w_h2_fl.flatten())
        axs[0].set_ylabel(r'$w_{\mathrm{H_2}, \infty}$')
        axs[0].set_xticklabels([])
        axs[0].xaxis.set_minor_locator(AutoMinorLocator(n=5))
        axs[0].yaxis.set_minor_locator(AutoMinorLocator(n=5))

        axs[1].plot(self.t, self.w_h2[0, :].flatten())
        axs[1].set_ylabel(r'$w_{\mathrm{H_2}, r = 0}$')
        axs[1].set_xlabel('Time / s')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator(n=5))
        axs[1].yaxis.set_minor_locator(AutoMinorLocator(n=5))

        fig.show()

    def plot_y_co2_fl_surf(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)

        axs[0].set_xlim(np.min(self.t), np.max(self.t))
        axs[1].set_xlim(np.min(self.t), np.max(self.t))

        axs[0].plot(self.t, self.y_co2_fl.flatten())
        axs[0].set_ylabel(r'$y_{\mathrm{CO_2}, \infty}$')
        axs[0].set_xticklabels([])
        axs[0].xaxis.set_minor_locator(AutoMinorLocator(n=5))
        axs[0].yaxis.set_minor_locator(AutoMinorLocator(n=5))

        axs[1].plot(self.t, self.y_co2[0, :].flatten())
        axs[1].set_ylabel(r'$y_{\mathrm{CO_2}, r = 0}$')
        axs[1].set_xlabel('t / s')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator(n=5))
        axs[1].yaxis.set_minor_locator(AutoMinorLocator(n=5))

        fig.show()

    def plot_y_h2_fl_surf(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)

        axs[0].set_xlim(np.min(self.t), np.max(self.t))
        axs[1].set_xlim(np.min(self.t), np.max(self.t))

        axs[0].plot(self.t, self.y_h2_fl.flatten())
        axs[0].set_ylabel(r'$y_{\mathrm{H_2}, \infty}$')
        axs[0].set_xticklabels([])
        axs[0].xaxis.set_minor_locator(AutoMinorLocator(n=5))
        axs[0].yaxis.set_minor_locator(AutoMinorLocator(n=5))

        axs[1].plot(self.t, self.y_h2[0, :].flatten())
        axs[1].set_ylabel(r'$y_{\mathrm{H_2}, r = 0}$')
        axs[1].set_xlabel('Time / s')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator(n=5))
        axs[1].yaxis.set_minor_locator(AutoMinorLocator(n=5))

        fig.show()

    def plot_frequency(self):
        # create figure
        fig = plt.figure(constrained_layout=False, figsize=(3.5, 3.5))
        ax = fig.add_subplot()

        y_max = max(np.max(self.T_fl), np.max(self.T[0, :]))
        y_min = min(np.min(self.T_fl), np.min(self.T[0, :]))
        y_min = y_min - 0.1 * (y_max - y_min)
        y_max = y_max + 0.1 * (y_max - y_min)

        ax.set_ylim(y_min, y_max)

        ax.set_ylabel(r'$T / \si{\K}$')
        ax.set_xlabel(r'$y_{\ce{H2}}$')
        ax.plot(self.y_h2[0, :], self.T[0, :], label='catalyst center', color=self.colors[0])
        ax.plot(self.y_h2[-1, :], self.T[-1, :], label='catalyst surface', color=self.colors[1])
        ax.plot(self.y_h2_fl.flatten(), self.T_fl.flatten(), label='feed', color=self.colors[2])
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        ax.legend(frameon=True, fancybox=False, ncol=1, loc='best', framealpha=1, edgecolor='black')

        fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/Frequency.pdf', pad_inches=self.fig_pad, bbox_inches='tight')

    def plot_frequency_center(self):
        # create figure
        fig = plt.figure(hash('freq_center'), constrained_layout=False, figsize=(6, 3))
        first = len(fig.axes) == 0
        last = False
        if first:
            ax = fig.add_subplot(1, 2, 1)
            ax.set_ylabel(r'$T / \si{\K}$')
            ax.set_xlabel(r'$y_{\ce{H2}}$')
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
            col = 0
        elif self.params.f_T == self.params.f_y:
            ax = fig.axes[0]
            col = 1
        elif len(fig.axes) == 1:
            ax = fig.add_subplot(1, 2, 2, sharey=fig.axes[0])
            ax.set_xlabel(r'$y_{\ce{H2}}$')
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
            ax.label_outer()
            col = 2
        else:
            ax = fig.axes[1]
            last = True
            col = 3

        ax.plot(self.y_h2[0, :], self.T[0, :], label=f'$f_y = f_T =  \SI{{{self.params.f_y}}}{{\per\s}}$' if self.params.f_T == self.params.f_y else f'$f_y = \SI{{{self.params.f_y}}}{{\per\s}} \quad f_T = \SI{{{self.params.f_T}}}{{\per\s}}$', color=self.colors[col])
        #ax.plot(self.y_h2_fl.flatten(), self.T_fl.flatten(), label='feed', color=self.colors[1])

        if last:
            fig.legend(frameon=True, fancybox=False, ncol=2, bbox_to_anchor=(0.8, 1.1), framealpha=1, edgecolor='black')
            fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/Frequency_center.pdf', pad_inches=self.fig_pad, bbox_inches='tight')

    def plot_dynamic_feed(self):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot()
        ax2 = ax.twinx()
        ax2.set_ylim(0,1)

        ax.plot(self.t, self.y_h2_fl.flatten(), label=r'$y_{\mathrm{H_2}}$')
        ax.plot(self.t, self.y_co2_fl.flatten(), label=r'$y_{\mathrm{CO_2}}$')
        ax2.plot(self.t, self.T_fl.flatten(), label=r'$T$')

        ax.set_ylabel(r'$y_i$')
        ax.set_xlabel(r'$y_\mathrm{H_2}$')
        ax2.set_ylabel(r'$T$ / K')

        fig.legend()
        fig.show()

    def plot_dynamic_feed_w(self):
        # create figure
        fig = plt.figure(hash('dynamic_feed_w'), constrained_layout=False, figsize=(6, 3))
        index = len(fig.axes) + 1
        ax = fig.add_subplot(1, 2, index, sharey=fig.axes[0] if index == 2 else None)

        rad = self.t * 2 * np.pi * self.params.f_y

        ax.axhline(self.params.w_h2_0, color=self.colors[7], alpha=0.4)
        ax.text(ca.pi / 8, self.params.w_h2_0 + 0.03, r'\ce{H2}', color=self.colors[0])
        ax.axhline(self.params.w_co2_0, color=self.colors[7], alpha=0.4)
        ax.text(ca.pi / 8, self.params.w_co2_0 - 0.07, r'\ce{CO2}', color=self.colors[1])

        ax.plot(rad, self.w_h2_fl.flatten(), color=self.colors[0])
        ax.plot(rad, self.w_co2_fl.flatten(), color=self.colors[1])

        ax.set_ylim(0, 1)
        ax.set_xlim(0, ca.pi * 2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

        ax.set_xlabel(r'radian')

        if index == 2:
            ax.text(-0.15, 1.06, '(b)', transform=ax.transAxes, fontsize=12, weight='bold')
            ax.label_outer()
            fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/DynamicFeed_w.pdf', pad_inches=self.fig_pad, bbox_inches='tight')
        else:
            ax.text(-0.25, 1.06, '(a)', transform=ax.transAxes, fontsize=12, weight='bold')
            ax.set_ylabel(r'$w_i$')

    def plot_dynamic_feed_T(self):
        # create figure
        fig = plt.figure(2, constrained_layout=False, figsize=(6, 3))
        index = len(fig.axes) + 1
        ax = fig.add_subplot(1, 2, index, sharey=fig.axes[0] if index == 2 else None)

        rad = self.t * 2 * np.pi * self.params.f_T

        ax.axhline(self.params.T_0, color=self.colors[7], alpha=0.4)

        ax.plot(rad, self.T_fl.flatten(), label=r'$T$', color=self.colors[2])

        #ax.set_ylim(515, 550)
        ax.set_xlim(0, ca.pi * 2)

        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

        ax.set_xlabel(r'radian')

        if index == 2:
            ax.text(-0.15, 1.06, '(b)', transform=ax.transAxes, fontsize=12, weight='bold')
            ax.label_outer()
            fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/DynamicFeed_T.pdf', pad_inches=self.fig_pad, bbox_inches='tight')
        else:
            ax.text(-0.25, 1.06, '(a)', transform=ax.transAxes, fontsize=12, weight='bold')
            ax.set_ylabel(r'$T$ / \si{\K}')

    def plot_closing_condition(self):
        # create figure
        fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, constrained_layout=False, figsize=(6, 3))

        t = len(self.params.t_i) - 1

        # set limits for x and y axis (r and w)
        ax.set_xlim(min(self.r), max(self.r))
        ax.set_ylim(0, 1)
        ax2.set_xlim(min(self.r), max(self.r))
        ax2.set_ylim(0, 1)

        ax.plot(self.r, self.w_co2[:, t], label=r'$w_{\ce{CO2}}$', color=self.colors[1], linestyle='solid')
        ax.plot(self.r, self.w_h2[:, t], label=r'$w_{\ce{H2}}$', color=self.colors[0], linestyle='dotted')
        ax.plot(self.r, self.w_ch4[:, t], label=r'$w_{\ce{CH4}}$', color=self.colors[2], linestyle='dashed')
        ax.plot(self.r, self.w_h2o[:, t], label=r'$w_{\ce{H2O}}$', color=self.colors[4], linestyle='dashdot')
        ax2.plot(self.r, np.load('/home/jona/PycharmProjects/CO2-Methanisation/h2_data_co2.npy'), label=r'$w_{\ce{CO2}}$', color=self.colors[1], linestyle='solid')
        ax2.plot(self.r, np.load('/home/jona/PycharmProjects/CO2-Methanisation/h2_data_h2.npy'), label=r'$w_{\ce{H2}}$', color=self.colors[0], linestyle='dotted')
        ax2.plot(self.r, np.load('/home/jona/PycharmProjects/CO2-Methanisation/h2_data_ch4.npy'), label=r'$w_{\ce{CH4}}$', color=self.colors[2], linestyle='dashed')
        ax2.plot(self.r, np.load('/home/jona/PycharmProjects/CO2-Methanisation/h2_data_h2o.npy'), label=r'$w_{\ce{H2O}}$', color=self.colors[4], linestyle='dashdot')

        # set title
        ax.set_xlabel('$r / \si{\mm}$')
        ax.set_ylabel(r'$w_i$')
        ax2.set_xlabel('$r / \si{\mm}$')
        #ax2.set_ylabel(r'$w_i$')
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax2.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(n=5))

        ax.text(-0.25,1.06, '(a)', transform=ax.transAxes, fontsize=12,  weight='bold')
        ax2.text(-0.15,1.06, '(b)', transform=ax2.transAxes, fontsize=12,  weight='bold')

        ax2.legend(frameon=True, fancybox=False, loc='best', ncol=1, framealpha=1, edgecolor='black')

        fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/Closing_Condition.pdf', pad_inches=self.fig_pad, bbox_inches='tight')

    def plot_conversion_per_temp(self):
        fig = plt.figure(hash('conversion_per_temp'), constrained_layout=False)
        if len(fig.axes) == 0:
            ax = fig.add_subplot()
            ax.set_xlabel(r'$T / \si{\K}$')
            ax.set_ylabel(r'$X_{\ce{CO2}}$')
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))

            ax.set_ylim(0, 1)
            ax.set_xlim(self.params.T_0, self.params.T_max)
        else:
            ax = fig.axes[0]
        ma = np.max(self.X_co2)
        if len(ax.lines) == 0:
            ax.plot(self.params.T_0, ma, color=self.colors[0])
        else:
            l = ax.lines[0]
            l.set_xdata(np.append(l.get_xdata(), self.params.T_0))
            l.set_ydata(np.append(l.get_ydata(), ma))

        if self.params.T_0 == self.params.T_max:
            #ax.relim()
            #ax.autoscale_view()
            fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/ConversionPerTemperature.pdf', pad_inches=self.fig_pad, bbox_inches='tight')

    def plot_w_i_feed_surface(self):
        fig = plt.figure(constrained_layout=False)
        ax = fig.add_subplot()

        ax.plot(self.t, self.w_co2[-1, :].flatten(), label=r'$w_{\ce{CO2}, \text{surf}}$')
        ax.plot(self.t, self.w_co2_fl.flatten(), label=r'$w_{\ce{CO2}, \text{f}}$')

        ax.legend(frameon=True, fancybox=False, loc='best', ncol=1, framealpha=1, edgecolor='black')
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/CompositionFeedSurface.pdf', pad_inches=self.fig_pad, bbox_inches='tight')

    def plot_interval(self):
        t = [0, 2/3*np.pi, 2/3*np.pi, np.pi, np.pi, 4/3*np.pi, 4/3*np.pi, 2*np.pi]

        # Define two different frequencies
        frequency1 = 1  # Frequency 1
        frequency2 = 1.5  # Frequency 2

        # Create the sinusoidal functions
        sinusoid1 = [1, 1, 1, 1, -1, -1, -1, -1]
        sinusoid2 = [0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5]

        # Plotting
        fig = plt.figure(constrained_layout=False)
        ax = fig.add_subplot()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.hlines(0, 0, 2 * np.pi, color=self.colors[7])
        ax.plot(t, sinusoid1, color=self.colors[0])
        ax.plot(t, sinusoid2, color=self.colors[1])
        ax.set_xlim(0, 2 * np.pi)

        ax.annotate('', xy=(2/3*np.pi, -0.05), xytext=(0, -0.05),
                    arrowprops=dict(arrowstyle='<->', color=self.colors[2], lw=2))
        ax.annotate('', xy=(np.pi, -0.05), xytext=(2/3*np.pi, -0.05),
                    arrowprops=dict(arrowstyle='<->', color=self.colors[2], lw=2))
        ax.annotate('', xy=(4/3*np.pi, -0.05), xytext=(np.pi, -0.05),
                    arrowprops=dict(arrowstyle='<->', color=self.colors[2], lw=2))
        ax.annotate('', xy=(2*np.pi, -0.05), xytext=(4/3*np.pi, -0.05),
                    arrowprops=dict(arrowstyle='<->', color=self.colors[2], lw=2))

        #ax.hline(-0.1, 0, 2 / 3 * np.pi, color=self.colors[7])

        fig.savefig('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/Intervals.pdf', pad_inches=self.fig_pad, bbox_inches='tight')
