import numpy as np

from diffusion import Diffusion
from parameters import Parameters
from plotter import Plotter
from reaction_rate import ReactionRate
import casadi as ca


class Integrator:
    def __init__(self, params: Parameters):
        self.params = params
        self.diff = Diffusion(self.params)
        self.r_rate = ReactionRate(self.params)

    def get_cf(self, T):
        # ideal gas
        return self.params.p_t * 1e5 / (self.params.R * T)  # (bar * 100000)(=Pa=J/m^3) / (J/(mol*K) * K) = mol/m^3

    def ode_y_i(self, y_i, y_i_surf, v_i, r, y_co2, y_h2, y_ch4, y_h2o):
         #return ((1 / self.params.epsilon) * self.diff.calc(y_i, y_i_surf, r)
          #       + ((1 / self.params.epsilon) - 1) * (self.params.roh_s / self.get_cf(self.params.T_ref))
           #     * v_i * self.r_rate.calc(y_co2, y_h2, y_ch4, y_h2o))
        return (((1 - self.params.epsilon) / self.params.epsilon) * (self.params.roh_s / self.get_cf(self.params.T_ref))
                * v_i * self.r_rate.calc(y_co2, y_h2, y_ch4, y_h2o))

    def run(self):
        a = ((1 - self.params.epsilon) / self.params.epsilon) * (self.params.roh_s / self.get_cf(self.params.T_ref)) * self.r_rate.calc(self.params.y_co2_0, self.params.y_h2_0, self.params.y_ch4_0, self.params.y_h2o_0)
        # create sym variables for y_i and T
        y_co2 = ca.SX.sym('y_co2', self.params.r_steps)
        y_h2 = ca.SX.sym('y_h2', self.params.r_steps)
        y_ch4 = ca.SX.sym('y_ch4', self.params.r_steps)
        y_h2o = ca.SX.sym('y_h2o', self.params.r_steps)
        T = ca.SX.sym('T', self.params.r_steps)

        # create ode variables
        ode_co2 = ca.SX.sym('ode_co2', self.params.r_steps)
        ode_h2 = ca.SX.sym('ode_h2', self.params.r_steps)
        ode_ch4 = ca.SX.sym('ode_ch4', self.params.r_steps)
        ode_h2o = ca.SX.sym('ode_h2o', self.params.r_steps)
        ode_T = ca.SX.sym('ode_T', self.params.r_steps)

        y_co2_surf = ca.SX.sym('y_co2_surf')
        y_h2_surf = ca.SX.sym('y_h2_surf')
        y_ch4_surf = ca.SX.sym('y_ch4_surf')
        y_h2o_surf = ca.SX.sym('y_h2o_surf')
        T_surf = ca.SX.sym('T_surf')

        # assign equations to ode for each radius r
        for r in range(self.params.r_steps):
            ode_co2[r] = self.ode_y_i(y_co2, y_co2_surf, self.params.v_co2, r, y_co2[r], y_h2[r], y_ch4[r], y_h2o[r])
            ode_h2[r] = self.ode_y_i(y_h2, y_h2_surf, self.params.v_h2, r, y_co2[r], y_h2[r], y_ch4[r], y_h2o[r])
            ode_ch4[r] = self.ode_y_i(y_ch4, y_ch4_surf, self.params.v_ch4, r, y_co2[r], y_h2[r], y_ch4[r], y_h2o[r])
            ode_h2o[r] = self.ode_y_i(y_h2o, y_h2o_surf, self.params.v_h2o, r, y_co2[r], y_h2[r], y_ch4[r], y_h2o[r])
            # ode_T[r] = ...

        # create integrator
        dae = {
            'x': ca.veccat(y_co2, y_h2, y_ch4, y_h2o),
            'p': ca.vertcat(y_co2_surf, y_h2_surf, y_ch4_surf, y_h2o_surf),
            'ode': ca.vertcat(ode_co2, ode_h2, ode_ch4, ode_h2o),
        }
        integrator = ca.integrator('I', 'idas', dae, 0, self.params.t_i)

        # create initial values
        y_co2_0 = np.full(self.params.r_steps, self.params.y_co2_0)
        y_h2_0 = np.full(self.params.r_steps, self.params.y_h2_0)
        y_ch4_0 = np.full(self.params.r_steps, self.params.y_ch4_0)
        y_h2o_0 = np.full(self.params.r_steps, self.params.y_h2o_0)
        T_0 = np.full(self.params.r_steps, self.params.T_0)
        x0 = ca.vertcat(y_co2_0, y_h2_0, y_ch4_0, y_h2o_0)

        # integrate
        res_dic = integrator(x0=x0, p=ca.vertcat(self.params.y_co2_0, self.params.y_h2_0, self.params.y_ch4_0,
                                                 self.params.y_h2o_0))
        res = ca.vertsplit(res_dic['xf'], self.params.r_steps)
        print(res)
        res_y_co2 = res[0]
        # res_T_i = res[1]
        plotter = Plotter()
        plotter.plot(self.params.t_i, np.linspace(0, 10, self.params.r_steps), res_y_co2.full())
