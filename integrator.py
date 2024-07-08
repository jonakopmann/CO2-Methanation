import numpy as np

from diffusion import Diffusion
from parameters import Parameters
from reaction_rate import ReactionRate
import casadi as ca


class Integrator:
    def __init__(self, params: Parameters):
        self.params = params
        self.diff = Diffusion(self.params)
        self.r_rate = ReactionRate(self.params)

    def run(self):
        # create sym variables for y_i and T
        y_i = ca.SX.sym('y_i', self.params.r_steps)
        T = ca.SX.sym('T', self.params.r_steps)

        # create ode variables
        ode_y = ca.SX.sym('ode_y', self.params.r_steps)
        ode_T = ca.SX.sym('ode_T', self.params.r_steps)

        # assign equations to ode for each radius r
        for r in range(self.params.r_steps):
            ode_y[r] = ((1 / self.params.epsilon) * self.diff.calc(y_i, r) + ((1 / self.params.epsilon) - 1) *
                        (self.params.roh_s / self.params.c_f) * self.params.v_i * self.r_rate.calc(y_i[r]))
            # ode_T[r] = ...

        # create integrator
        # dae = {'x': ca.vertcat(y_i, T), 'ode': ca.vertcat(ode_y, ode_T)}
        dae = {'x': y_i, 'ode': ode_y}  # isotherm
        integrator = ca.integrator('I', 'idas', dae, 0, self.params.t_i)

        # create initial values
        y_i_0 = np.full(self.params.r_steps, self.params.y_i_0)
        T_0 = np.full(self.params.r_steps, self.params.T_0)
        # x0 = ca.vertcat(y_i_0, T_0)
        x0 = y_i_0  # isotherm

        # integrate
        res_dic = integrator(x0=x0)
        # res = ca.vertsplit(res_dic['xf'], self.params.steps)
        # res_y_i = res[0]
        # res_T_i = res[1]
        res_y_i = res_dic['xf']  # isotherm
        print(res_y_i)
