import numpy as np

from diffusion import Diffusion
from heat_conduction import HeatConduction
from parameters import Parameters
from plotter import Plotter
from reaction_rate import ReactionRate
import casadi as ca


class Integrator:
    def __init__(self, params: Parameters):
        self.params = params
        self.diff = Diffusion(self.params)
        self.r_rate = ReactionRate(self.params)
        self.h_conduction = HeatConduction(self.params)

    def get_cf(self, T):
        # ideal gas
        return self.params.p_t * 1e5 / (self.params.R * T)  # (bar * 100000)(=Pa=J/m^3) / (J/(mol*K) * K) = mol/m^3

    def ode_y_i(self, y_i, y_i_surf, v_i, i, r, T):
        return ((1 / self.params.epsilon) * self.diff.calc(y_i, y_i_surf, i)
                + ((1 / self.params.epsilon) - 1) * (self.params.roh_s / self.get_cf(T))
                * v_i * r)

    def run(self):
        # create sym variables for y_i, T and t
        y_co2 = ca.SX.sym('y_co2', self.params.r_steps)
        y_h2 = ca.SX.sym('y_h2', self.params.r_steps)
        y_ch4 = ca.SX.sym('y_ch4', self.params.r_steps)
        y_h2o = ca.SX.sym('y_h2o', self.params.r_steps)
        T = ca.SX.sym('T', self.params.r_steps)
        t = ca.SX.sym('t')

        # create ode variables
        ode_co2 = ca.SX.sym('ode_co2', self.params.r_steps)
        ode_h2 = ca.SX.sym('ode_h2', self.params.r_steps)
        ode_ch4 = ca.SX.sym('ode_ch4', self.params.r_steps)
        ode_h2o = ca.SX.sym('ode_h2o', self.params.r_steps)
        ode_T = ca.SX.sym('ode_T', self.params.r_steps)

        # create surface variables
        y_co2_surf = ca.SX.sym('y_co2_surf')
        y_h2_surf = ca.SX.sym('y_h2_surf')
        y_ch4_surf = ca.SX.sym('y_ch4_surf')
        y_h2o_surf = ca.SX.sym('y_h2o_surf')
        T_surf = ca.SX.sym('T_surf')

        # create alg equations for the surface values
        alg_co2 = (self.params.y_co2_0
                   + self.params.delta_y * ca.sin(2 * ca.pi * self.params.f_y * t) - y_co2_surf)
        alg_h2 = (self.params.y_h2_0
                  - self.params.delta_y * ca.sin(2 * ca.pi * self.params.f_y * t) - y_h2_surf)
        alg_ch4 = (self.params.y_ch4_0
                   + 0 * self.params.delta_y * ca.sin(2 * ca.pi * self.params.f_y * t) - y_ch4_surf)
        alg_h2o = (self.params.y_h2o_0
                   + 0 * self.params.delta_y * ca.sin(2 * ca.pi * self.params.f_y * t) - y_h2o_surf)
        alg_T = self.params.T_0 + self.params.delta_T * ca.sin(2 * ca.pi * self.params.f_T * t) - T_surf

        # z variables
        r = ca.SX.sym('r', self.params.r_steps)
        alg_r = ca.SX.sym('alg_r', self.params.r_steps)

        # assign equations to ode for each radius i
        for i in range(self.params.r_steps):
            ode_co2[i] = self.ode_y_i(y_co2, y_co2_surf, self.params.v_co2, i, r[i], T[i])
            ode_h2[i] = self.ode_y_i(y_h2, y_h2_surf, self.params.v_h2, i, r[i], T[i])
            ode_ch4[i] = self.ode_y_i(y_ch4, y_ch4_surf, self.params.v_ch4, i, r[i], T[i])
            ode_h2o[i] = self.ode_y_i(y_h2o, y_h2o_surf, self.params.v_h2o, i, r[i], T[i])
            ode_T[i] = (1e9 / (self.params.roh_s * self.params.c_p) * self.h_conduction.calc(T, T_surf, i)
                        - (1 - self.params.epsilon) / self.params.c_p * self.r_rate.get_H_R(T[i])
                        * r[i])
            alg_r[i] = self.r_rate.calc(y_co2[i], y_h2[i], y_ch4[i], y_h2o[i], T[i]) - r[i]

        # create integrator
        dae = {
            'x': ca.veccat(y_co2, y_h2, y_ch4, y_h2o, T),
            'z': ca.vertcat(y_co2_surf, y_h2_surf, y_ch4_surf, y_h2o_surf, T_surf, r),
            't': t,
            'ode': ca.vertcat(ode_co2, ode_h2, ode_ch4, ode_h2o, ode_T),
            'alg': ca.vertcat(alg_co2, alg_h2, alg_ch4, alg_h2o, alg_T, alg_r)
        }
        integrator = ca.integrator('I', 'idas', dae, 0, self.params.t_i)

        # create initial values
        y_co2_0 = np.full(self.params.r_steps, self.params.y_co2_0)
        y_h2_0 = np.full(self.params.r_steps, self.params.y_h2_0)
        y_ch4_0 = np.full(self.params.r_steps, self.params.y_ch4_0)
        y_h2o_0 = np.full(self.params.r_steps, self.params.y_h2o_0)
        T_0 = np.full(self.params.r_steps, self.params.T_0)
        x0 = ca.vertcat(y_co2_0, y_h2_0, y_ch4_0, y_h2o_0, T_0)

        # integrate
        res = integrator(x0=x0)
        res_x = ca.vertsplit(res['xf'], self.params.r_steps)
        res_z = res['zf']
        # add surface values
        res_y_co2 = ca.vertcat(res_x[0], res_z[0, :])
        res_y_h2 = ca.vertcat(res_x[1], res_z[1, :])
        res_y_ch4 = ca.vertcat(res_x[2], res_z[2, :])
        res_y_h2o = ca.vertcat(res_x[3], res_z[3, :])
        res_T = ca.vertcat(res_x[4], res_z[4, :])

        # plot y_i and T
        plotter = Plotter()
        plotter.plot(self.params.t_i, np.linspace(0, self.params.r_max, self.params.r_steps + 1), res_y_co2.full(),
                     't / s', 'r / mm', r'$y_\mathrm{CO_2}$', r'$\mathrm{CO_2}$', 0, 1)
        plotter.plot(self.params.t_i, np.linspace(0, self.params.r_max, self.params.r_steps + 1), res_y_h2.full(),
                     't / s', 'r / mm', r'$y_\mathrm{H_2}$', r'$\mathrm{H_2}$', 0, 1)
        plotter.plot(self.params.t_i, np.linspace(0, self.params.r_max, self.params.r_steps + 1), res_y_ch4.full(),
                     't / s', 'r / mm', r'$y_\mathrm{CH_4}$', r'$\mathrm{CH_4}$', 0, 1)
        plotter.plot(self.params.t_i, np.linspace(0, self.params.r_max, self.params.r_steps + 1), res_y_h2o.full(),
                     't / s', 'r / mm', r'$y_\mathrm{H_2O}$', r'$\mathrm{H_2O}$', 0, 1)
        plotter.plot(self.params.t_i, np.linspace(0, self.params.r_max, self.params.r_steps + 1), res_T.full(),
                     't / s', 'r / mm', 'T / K', 'Temperature')
