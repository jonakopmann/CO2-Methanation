import os

import numpy as np

from diffusion import Diffusion
from heat_conduction import HeatConduction
from parameters import Parameters
from plotter import Plotter
from reaction import Reaction
from thermo import *
import casadi as ca


class Integrator:
    def __init__(self, params: Parameters, debug=False):
        self.params = params
        self.diff = Diffusion(self.params)
        self.reaction = Reaction(self.params)
        self.heat_cond = HeatConduction(self.params)
        self.debug = debug

    def get_D_i_j(self, T, p, M_i, M_j, v_i, v_j):
        return 1e-1 * (T ** 1.75) * (((M_i ** -1) + (M_j ** -1)) ** 0.5) / (
                p * 0.98692327 * ((v_i ** (1 / 3)) + (v_j ** (1 / 3))) ** 2)

    def get_D_i_m(self, y_i, y_j_1, y_j_2, y_j_3, D_i_j_1, D_i_j_2, D_i_j_3):
        return (1 - y_i) / ((y_j_1 / D_i_j_1) + (y_j_2 / D_i_j_2) + (y_j_3 / D_i_j_3))

    def get_D_i_Kn(self, T, M_i):
        return self.params.d_pore / 3 * ca.sqrt(8e3 * self.params.R * T / (ca.pi * M_i))

    def run(self):
        slices = int(self.params.f_y * self.params.t_max)
        self.params.t_steps = int((self.params.t_steps + 1) / slices) * slices
        self.params.t_i = np.linspace(0, self.params.t_max, self.params.t_steps)
        step_size = self.params.t_steps / slices
        res_final = {'xf': [], 'zf': []}
        for k in range(slices):
            # create sym variables for y_i, T and t
            w_co2 = ca.SX.sym('w_co2', self.params.r_steps)
            w_ch4 = ca.SX.sym('w_ch4', self.params.r_steps)
            w_h2o = ca.SX.sym('w_h2o', self.params.r_steps)
            w_h2 = 1 - w_co2 - w_ch4 - w_h2o
            T = ca.SX.sym('T', self.params.r_steps)
            t = ca.SX.sym('t')
            p = ca.SX.sym('p', self.params.r_steps)

            # create ode variables
            ode_co2 = ca.SX.sym('ode_co2', self.params.r_steps)
            ode_ch4 = ca.SX.sym('ode_ch4', self.params.r_steps)
            ode_h2o = ca.SX.sym('ode_h2o', self.params.r_steps)
            ode_T = ca.SX.sym('ode_T', self.params.r_steps)
            alg_p = ca.SX.sym('alg_p', self.params.r_steps)

            # create surface variables
            w_co2_surf = ca.SX.sym('w_co2_surf')
            w_co2_fl = ca.SX.sym('w_co2_fl')
            w_ch4_surf = ca.SX.sym('w_ch4_surf')
            w_ch4_fl = ca.SX.sym('w_ch4_fl')
            w_h2o_surf = ca.SX.sym('w_h20_surf')
            w_h2o_fl = ca.SX.sym('w_h2o_fl')
            T_surf = ca.SX.sym('T_surf')
            T_fl = ca.SX.sym('T_fl')
            w_h2_surf = 1 - w_co2_surf - w_ch4_surf - w_h2o_surf
            w_h2_fl = 1 - w_co2_fl - w_ch4_fl - w_h2o_fl

            D_co2_eff = ca.SX.sym('D_co2_eff', self.params.r_steps)
            alg_D_co2 = ca.SX.sym('alg_D_co2', self.params.r_steps)
            D_ch4_eff = ca.SX.sym('D_ch4_eff', self.params.r_steps)
            alg_D_ch4 = ca.SX.sym('alg_D_ch4', self.params.r_steps)
            D_h2o_eff = ca.SX.sym('D_h2o_eff', self.params.r_steps)
            alg_D_h2o = ca.SX.sym('alg_D_h2o', self.params.r_steps)

            M_fl = (w_co2_fl / self.params.M_co2 + w_h2_fl / self.params.M_h2
                    + w_ch4_fl / self.params.M_ch4 + w_h2o_fl / self.params.M_h2o) ** -1
            M_surf = (w_co2_surf / self.params.M_co2 + w_h2_surf / self.params.M_h2
                      + w_ch4_surf / self.params.M_ch4 + w_h2o_surf / self.params.M_h2o) ** -1
            M_1 = (w_co2[-1] / self.params.M_co2 + w_h2[-1] / self.params.M_h2
                   + w_ch4[-1] / self.params.M_ch4 + w_h2o[-1] / self.params.M_h2o) ** -1

            roh_fl = p[-1] * 1e5 * M_fl / (self.params.R * T_fl)  # [g/m^3]
            roh_surf = p[-1] * 1e5 * M_surf / (self.params.R * T_surf)
            roh_1 = p[-1] * 1e5 * M_1 / (self.params.R * T[-1])

            cp_fl = (w_co2_fl * get_cp_co2(T_fl) + w_h2_fl * get_cp_h2(T_fl)
                     + w_ch4_fl * get_cp_ch4(T_fl) + w_h2o_fl * get_cp_h2o(T_fl))  # [J/(g*K)]
            ny_fl = (w_co2_fl * get_ny_co2(T_fl, p[-1]) + w_h2_fl * get_ny_h2(T_fl, p[-1])
                     + w_ch4_fl * get_ny_ch4(T_fl, p[-1]) + w_h2o_fl * get_ny_h2o(T_fl, p[-1]))  # [mm^2/s]

            lambda_fl = (w_co2_fl * get_lambda_co2(T_fl) + w_h2_fl * get_lambda_h2(T_fl)
                         + w_ch4_fl * get_lambda_ch4(T_fl) + w_h2o_fl * get_lambda_h2o(T_fl))  # [W/(mm*K)]

            Re = self.params.v * self.params.r_max * 2 / ny_fl
            Pr = ny_fl / lambda_fl * cp_fl * roh_fl * 1e-9
            Nu = 2 + 0.6 * Re ** 0.5 + Pr ** (1 / 3)
            alpha = Nu * lambda_fl / (2 * self.params.r_max)  # [W/(mm^2*K)]

            # species transfer
            def get_beta_i(D_i_eff):
                Sc = ny_fl / D_i_eff
                Sh = 2 + 0.6 * Re ** 0.5 + Sc ** (1 / 3)
                return Sh * D_i_eff / (2 * self.params.r_max)  # [mm/s]

            # create alg equations for the surface values
            alg_co2_surf = (w_co2_fl * roh_fl - D_co2_eff[-1] / get_beta_i(D_co2_eff[-1])
                            * (w_co2_surf * roh_surf - w_co2[-1] * roh_1) / self.params.h - w_co2_surf * roh_surf)
            alg_ch4_surf = (w_ch4_fl * roh_fl / self.params.M_ch4 - D_ch4_eff[-1] / get_beta_i(D_ch4_eff[-1])
                            * (w_ch4_surf * roh_surf - w_ch4[-1] * roh_1) / self.params.h - w_ch4_surf * roh_surf)
            alg_h2o_surf = (w_h2o_fl * roh_fl - D_h2o_eff[-1] / get_beta_i(D_h2o_eff[-1])
                            * (w_h2o_surf * roh_surf - w_h2o[-1] * roh_1) / self.params.h - w_h2o_surf * roh_surf)

            alg_T_surf = T_fl - (self.params.lambda_eff / alpha * (T_surf - T[-1]) / self.params.h) - T_surf

            # assign equations to ode for each radius i
            for i in range(self.params.r_steps):
                # molar mass, reaction rate and density
                M = (w_co2[i] / self.params.M_co2 + w_h2[i] / self.params.M_h2
                     + w_ch4[i] / self.params.M_ch4 + w_h2o[i] / self.params.M_h2o) ** -1
                r = self.reaction.get_r(w_co2[i], w_h2[i], w_ch4[i], w_h2o[i], T[i], p[i], M)
                roh_g = p[i] * 1e5 * M / (self.params.R * T[i])

                ode_co2[i] = (self.diff.get_term(w_co2, w_co2_surf, i, D_co2_eff[i])
                              + self.reaction.get_mass_term(self.params.M_co2, roh_g, self.params.v_co2, r))
                ode_ch4[i] = (self.diff.get_term(w_ch4, w_ch4_surf, i, D_ch4_eff[i])
                              + self.reaction.get_mass_term(self.params.M_ch4, roh_g, self.params.v_ch4, r))
                ode_h2o[i] = (self.diff.get_term(w_h2o, w_h2o_surf, i, D_h2o_eff[i])
                              + self.reaction.get_mass_term(self.params.M_h2o, roh_g, self.params.v_h2o, r))
                ode_T[i] = (self.heat_cond.get_term(T, T_surf, i)
                            + self.reaction.get_heat_term(T[i], r))
                alg_p[i] = (self.params.M_0 * T[i]) / (M * self.params.T_0) * self.params.p_0 - p[i]

                D_co2_h2 = self.get_D_i_j(T[i], p[i], self.params.M_co2, self.params.M_h2, self.params.delta_v_co2,
                                          self.params.delta_v_h2)
                D_co2_ch4 = self.get_D_i_j(T[i], p[i], self.params.M_co2, self.params.M_ch4, self.params.delta_v_co2,
                                           self.params.delta_v_ch4)
                D_co2_h2o = self.get_D_i_j(T[i], p[i], self.params.M_co2, self.params.M_h2o, self.params.delta_v_co2,
                                           self.params.delta_v_h2o)
                D_h2_h2o = self.get_D_i_j(T[i], p[i], self.params.M_h2, self.params.M_h2o, self.params.delta_v_h2,
                                          self.params.delta_v_h2o)
                D_h2_ch4 = self.get_D_i_j(T[i], p[i], self.params.M_h2, self.params.M_ch4, self.params.delta_v_h2,
                                          self.params.delta_v_ch4)
                D_ch4_h2o = self.get_D_i_j(T[i], p[i], self.params.M_ch4, self.params.M_h2o, self.params.delta_v_ch4,
                                           self.params.delta_v_h2o)

                y_co2 = w_to_y(w_co2[i], self.params.M_co2, M)
                y_h2 = w_to_y(w_h2[i], self.params.M_h2, M)
                y_ch4 = w_to_y(w_ch4[i], self.params.M_ch4, M)
                y_h2o = w_to_y(w_h2o[i], self.params.M_h2o, M)

                alg_D_co2[i] = ((self.params.epsilon / self.params.tau
                                 * (1 / self.get_D_i_m(y_co2, y_h2, y_ch4, y_h2o, D_co2_h2, D_co2_ch4, D_co2_h2o)
                                    + 1e-6 / self.get_D_i_Kn(T[i], self.params.M_co2)) ** -1) - D_co2_eff[i])
                alg_D_ch4[i] = ((self.params.epsilon / self.params.tau
                                 * (1 / self.get_D_i_m(y_ch4, y_h2, y_co2, y_h2o, D_h2_ch4, D_co2_ch4, D_ch4_h2o)
                                    + 1e-6 / self.get_D_i_Kn(T[i], self.params.M_ch4)) ** -1) - D_ch4_eff[i])
                alg_D_h2o[i] = ((self.params.epsilon / self.params.tau
                                 * (1 / self.get_D_i_m(y_h2o, y_h2, y_co2, y_ch4, D_h2_h2o, D_co2_h2o, D_ch4_h2o)
                                    + 1e-6 / self.get_D_i_Kn(T[i], self.params.M_h2o)) ** -1) - D_h2o_eff[i])

            if k & 1 == 0:
                alg_co2_fl = (self.params.w_co2_0 + self.params.delta_w - w_co2_fl)
                alg_T_fl = self.params.T_0 + self.params.delta_T - T_fl
            else:
                alg_co2_fl = (self.params.w_co2_0 - self.params.delta_w - w_co2_fl)
                alg_T_fl = self.params.T_0 - self.params.delta_T - T_fl
            alg_ch4_fl = (self.params.w_ch4_0
                          + 0 * self.params.delta_w * ca.sin(2 * ca.pi * self.params.f_y * t) - w_ch4_fl)
            alg_h2o_fl = (self.params.w_h2o_0
                          + 0 * self.params.delta_w * ca.sin(2 * ca.pi * self.params.f_y * t) - w_h2o_fl)

            # create integrator
            dae = {
                'x': ca.veccat(w_co2, w_ch4, w_h2o, T),
                'z': ca.vertcat(w_co2_surf, w_ch4_surf, w_h2o_surf, T_surf,
                                w_co2_fl, w_ch4_fl, w_h2o_fl, T_fl,
                                D_co2_eff, D_ch4_eff, D_h2o_eff, p),
                't': t,
                'ode': ca.vertcat(ode_co2, ode_ch4, ode_h2o, ode_T),
                'alg': ca.vertcat(alg_co2_surf, alg_ch4_surf, alg_h2o_surf, alg_T_surf,
                                  alg_co2_fl, alg_ch4_fl, alg_h2o_fl, alg_T_fl,
                                  alg_D_co2, alg_D_ch4, alg_D_h2o, alg_p)
            }

            options = {'regularity_check': False}
            if self.debug:
                options['verbose'] = True
                options['monitor'] = 'daeF'

            integrator = ca.integrator('I', 'idas', dae, k * self.params.t_max / slices,
                                       self.params.t_i[int(k * step_size):int(k * step_size + step_size)], options)

            # create initial values
            if k == 0:
                w_co2_0 = np.full(self.params.r_steps, self.params.w_co2_0)
                w_ch4_0 = np.full(self.params.r_steps, self.params.w_ch4_0)
                w_h2o_0 = np.full(self.params.r_steps, self.params.w_h2o_0)
                T_0 = np.full(self.params.r_steps, self.params.T_0)
                x0 = ca.vertcat(w_co2_0, w_ch4_0, w_h2o_0, T_0)

                z0 = ca.vertcat(self.params.w_co2_0, self.params.w_ch4_0, self.params.w_h2o_0,
                                self.params.T_0, self.params.w_co2_0, self.params.w_ch4_0, self.params.w_h2o_0,
                                self.params.T_0,
                                np.full(self.params.r_steps, 1), np.full(self.params.r_steps, 1),
                                np.full(self.params.r_steps, 1), np.full(self.params.r_steps, 1))
            else:
                x0 = res_final['xf'][:, -1]
                z0 = res_final['zf'][:, -1]

            # integrate
            res = integrator(x0=x0, z0=z0)
            res_final['xf'] = ca.horzcat(res_final['xf'], res['xf'])
            res_final['zf'] = ca.horzcat(res_final['zf'], res['zf'])

        res_x = ca.vertsplit(res_final['xf'], self.params.r_steps)
        res_z = ca.vertsplit(res_final['zf'])

        # add surface values
        res_w_co2 = ca.vertcat(res_x[0], res_z[0])
        res_w_ch4 = ca.vertcat(res_x[1], res_z[1])
        res_w_h2o = ca.vertcat(res_x[2], res_z[2])
        res_w_h2 = 1 - res_w_co2 - res_w_ch4 - res_w_h2o
        res_T = ca.vertcat(res_x[3], res_z[3])
        res_p = ca.vertcat(ca.vertcat(*res_z[-self.params.r_steps:]), res_z[-self.params.r_steps:][-1])
        res_err = 1 - res_w_co2.full() - res_w_h2.full() - res_w_ch4.full() - res_w_h2o.full()

        # print max error
        print(f'max error {np.max(res_err)}')

        # create plotter and plot
        plotter = Plotter(self.params.t_i, np.linspace(0, self.params.r_max, self.params.r_steps + 1),
                          res_w_co2.full(), res_w_h2.full(), res_w_ch4.full(), res_w_h2o.full(), res_T.full(),
                          res_p.full())

        path = f'plots_nu/f-{self.params.f_y}_deltaw-{self.params.delta_w}_deltaT-{self.params.delta_T}_t-{self.params.t_max}'
        plotter.animate_w(os.path.join(path, 'weight.mp4'), 'Mass fractions over time', 1)
        plotter.animate_T(os.path.join(path, 'temp.mp4'), 'Temperature over time', 1, )
        # plotter.animate_Y(os.path.join(path, 'yield.mp4'), 'Yield over time', 1)
        # plotter.animate_y(os.path.join(path, 'mole.mp4'), 'Mole fractions over time', 1)
