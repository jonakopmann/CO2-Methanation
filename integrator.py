import math
import os

import casadi as ca
import numpy as np

from context import Context
from diffusion import Diffusion
from diffusion_coefficient import DiffusionCoefficient
from heat_conduction import HeatConduction
from parameters import Parameters
from plotter import Plotter
from reaction import Reaction
from slice import Slice


def frequency_intervals(t, f1, f2):
    # TODO: make this cleaner
    # this is straight from ai, but it works so its ok for now

    # Calculate time intervals for each frequency
    interval_f1 = 1 / f1
    interval_f2 = 1 / f2

    # Initialize arrays to store time intervals
    intervals_f1 = []
    intervals_f2 = []

    # Calculate time intervals for each cycle of f1
    for i in range(int(t / interval_f1) + 1):
        start = i * interval_f1
        end = (i + 1) * interval_f1
        if end > t:
            end = t
        intervals_f1.append((start, end))

    # Calculate time intervals for each cycle of f2
    for i in range(int(t / interval_f2) + 1):
        start = i * interval_f2
        end = (i + 1) * interval_f2
        if end > t:
            end = t
        intervals_f2.append((start, end))

    # Combine intervals
    combined_intervals = []
    for start_f1, end_f1 in intervals_f1:
        for start_f2, end_f2 in intervals_f2:
            if start_f1 < end_f2 and start_f2 < end_f1:
                combined_intervals.append((max(start_f1, start_f2), min(end_f1, end_f2)))

    # Remove duplicates and sort intervals
    combined_intervals = sorted(list(set(combined_intervals)))

    # Convert intervals to durations
    durations = [combined_intervals[i][1] - combined_intervals[i][0] for i in range(len(combined_intervals))]

    return durations


class Integrator:
    def __init__(self, params: Parameters, debug=False):
        self.params = params
        self.debug = debug

    def create_t_i(self):
        # we need to create a time array that has all the intersections from our 2 frequencies
        intervals = frequency_intervals(self.params.t_max, self.params.f_w, self.params.f_T)

        t0 = 0
        t = [np.array([0.0])]
        slices = []
        index = 0
        w_1 = 0
        T_1 = 0
        w = False
        T = False
        for i in range(len(intervals)):
            if ca.fabs(t0 * self.params.f_w - w_1) <= 1e-10:
                w = not w
                w_1 += 1
            if ca.fabs(t0 * self.params.f_T - T_1) <= 1e-10:
                T = not T
                T_1 += 1
            steps = int(max(self.params.fps * intervals[i], self.params.x_min))
            t.append(np.delete(np.linspace(t0, t0 + intervals[i], steps), 0))
            t0 = t0 + intervals[i]
            slices.append(Slice(w, T, index, steps))
            index += steps - 1

        self.params.t_i = np.concatenate(t)
        return slices

    def run(self):
        # create our intervals and final output dict
        slices = self.create_t_i()
        res_final = {'xf': [], 'zf': []}

        for k in slices:
            # create context
            ctx = Context(self.params)
            diff = Diffusion(self.params)
            diff_i_eff = DiffusionCoefficient(ctx)
            reaction = Reaction(ctx)
            heat_cond = HeatConduction(ctx)

            # create ode variables
            ode_co2 = ca.SX(self.params.r_steps, 1)
            ode_ch4 = ca.SX(self.params.r_steps, 1)
            ode_h2 = ca.SX(self.params.r_steps, 1)
            ode_T = ca.SX(self.params.r_steps, 1)
            alg_p = ca.SX(self.params.r_steps, 1)

            alg_D_co2 = ca.SX(self.params.r_steps, 1)
            alg_D_ch4 = ca.SX(self.params.r_steps, 1)
            alg_D_h2 = ca.SX(self.params.r_steps, 1)

            # create boundary conditions for the surface values
            alg_co2_surf = (ctx.w_co2_fl * ctx.rho_fl - (ctx.D_co2_eff[-1] / ctx.beta_co2
                            * (ctx.w_co2_surf * ctx.rho_surf - ctx.w_co2[-1] * ctx.rho[-1]) / self.params.h)
                            - ctx.w_co2_surf * ctx.rho_surf)
            alg_ch4_surf = (ctx.w_ch4_fl * ctx.rho_fl / self.params.M_ch4 - ctx.D_ch4_eff[-1] / ctx.beta_ch4
                            * (ctx.w_ch4_surf * ctx.rho_surf - ctx.w_ch4[-1] * ctx.rho[-1]) / self.params.h
                            - ctx.w_ch4_surf * ctx.rho_surf)
            alg_h2_surf = (ctx.w_h2_fl * ctx.rho_fl - ctx.D_h2_eff[-1] / ctx.beta_h2
                           * (ctx.w_h2_surf * ctx.rho_surf - ctx.w_h2[-1] * ctx.rho[-1]) / self.params.h
                           - ctx.w_h2_surf * ctx.rho_surf)
            alg_T_surf = (ctx.T_fl - (self.params.lambda_eff / ctx.alpha * (ctx.T_surf - ctx.T[-1]) / self.params.h)
                          - ctx.T_surf)

            # make input dynamic
            if k.w:
                alg_co2_fl = (self.params.w_co2_0 + self.params.delta_w - ctx.w_co2_fl)
                alg_h2_fl = (self.params.w_h2_0 - self.params.delta_w - ctx.w_h2_fl)
            else:
                alg_co2_fl = (self.params.w_co2_0 - self.params.delta_w - ctx.w_co2_fl)
                alg_h2_fl = (self.params.w_h2_0 + self.params.delta_w - ctx.w_h2_fl)
            if k.T:
                alg_T_fl = self.params.T_0 + self.params.delta_T - ctx.T_fl
            else:
                alg_T_fl = self.params.T_0 - self.params.delta_T - ctx.T_fl
            alg_ch4_fl = (self.params.w_ch4_0 - ctx.w_ch4_fl)

            # assign equations to ode for each radius i
            for i in range(self.params.r_steps):
                # get reaction rate
                r = reaction.get_r(i)

                # odes for w, T and p
                ode_co2[i] = (diff.get_term(ctx.w_co2, ctx.w_co2_surf, i, ctx.D_co2_eff[i])
                              + reaction.get_mass_term(self.params.M_co2, ctx.rho[i], self.params.v_co2, r))
                ode_ch4[i] = (diff.get_term(ctx.w_ch4, ctx.w_ch4_surf, i, ctx.D_ch4_eff[i])
                              + reaction.get_mass_term(self.params.M_ch4, ctx.rho[i], self.params.v_ch4, r))
                ode_h2[i] = (diff.get_term(ctx.w_h2, ctx.w_h2_surf, i, ctx.D_h2_eff[i])
                             + reaction.get_mass_term(self.params.M_h2, ctx.rho[i], self.params.v_h2, r))
                ode_T[i] = heat_cond.get_term(i) + reaction.get_heat_term(ctx.T[i], r)
                alg_p[i] = (self.params.M_0 * ctx.T[i]) / (ctx.M[i] * self.params.T_0) * self.params.p_0 - ctx.p[i]

                # init binary diffusion coefficients
                diff_i_eff.init(i)

                # alg for D_i_eff
                alg_D_co2[i] = diff_i_eff.get_D_co2(i) - ctx.D_co2_eff[i]
                alg_D_h2[i] = diff_i_eff.get_D_h2(i) - ctx.D_h2_eff[i]
                alg_D_ch4[i] = diff_i_eff.get_D_ch4(i) - ctx.D_ch4_eff[i]

            # create integrator
            dae = {
                'x': ca.veccat(ctx.w_co2, ctx.w_ch4, ctx.w_h2, ctx.T),
                'z': ca.vertcat(ctx.w_co2_surf, ctx.w_ch4_surf, ctx.w_h2_surf, ctx.T_surf,
                                ctx.w_co2_fl, ctx.w_ch4_fl, ctx.w_h2_fl, ctx.T_fl,
                                ctx.D_co2_eff, ctx.D_ch4_eff, ctx.D_h2_eff, ctx.p),
                't': ctx.t,
                'ode': ca.vertcat(ode_co2, ode_ch4, ode_h2, ode_T),
                'alg': ca.vertcat(alg_co2_surf, alg_ch4_surf, alg_h2_surf, alg_T_surf,
                                  alg_co2_fl, alg_ch4_fl, alg_h2_fl, alg_T_fl,
                                  alg_D_co2, alg_D_ch4, alg_D_h2, alg_p)
            }

            options = {'regularity_check': True}
            if self.debug:
                options['verbose'] = True
                options['monitor'] = 'daeF'

            t0 = float(self.params.t_i[k.step])
            t = self.params.t_i[k.step:k.step + k.step_size]
            integrator = ca.integrator('I', 'idas', dae, t0,
                                       t, options)

            # create initial values
            if k.step == 0:
                w_co2_0 = np.full(self.params.r_steps, self.params.w_co2_0)
                w_ch4_0 = np.full(self.params.r_steps, self.params.w_ch4_0)
                w_h2_0 = np.full(self.params.r_steps, self.params.w_h2_0)
                T_0 = np.full(self.params.r_steps, self.params.T_0)
                x0 = ca.vertcat(w_co2_0, w_ch4_0, w_h2_0, T_0)

                z0 = ca.vertcat(self.params.w_co2_0, self.params.w_ch4_0, self.params.w_h2_0, self.params.T_0,
                                self.params.w_co2_0, self.params.w_ch4_0, self.params.w_h2_0, self.params.T_0,
                                np.full(self.params.r_steps, 1), np.full(self.params.r_steps, 1),
                                np.full(self.params.r_steps, 1), np.full(self.params.r_steps, 1))
            else:
                x0 = res_final['xf'][:, -1]
                z0 = res_final['zf'][:, -1]

            # integrate
            res = integrator(x0=x0, z0=z0)
            skip = 0
            if k.step != 0:
                skip = 1
            res_final['xf'] = ca.horzcat(res_final['xf'], res['xf'][:, skip:])
            res_final['zf'] = ca.horzcat(res_final['zf'], res['zf'][:, skip:])

        # create plotter and plot
        plotter = Plotter(self.params.t_i, np.linspace(0, self.params.r_max, self.params.r_steps + 1), res_final, self.params)

        path = f'plots_water/fw-{self.params.f_w}_deltaw-{self.params.delta_w}_fT-{self.params.f_T}_deltaT-{self.params.delta_T}_t-{self.params.t_max}'
        #plotter.animate_X_co2('')
        plotter.animate_w(os.path.join(path, 'weight.mp4'), 'Mass fractions over time', 1)
        # plotter.animate_T(os.path.join(path, 'temp.mp4'), 'Temperature over time', 1, )
        # plotter.animate_Y(os.path.join(path, 'yield.mp4'), 'Yield over time', 1)
        plotter.animate_y(os.path.join(path, 'mole.mp4'), 'Mole fractions over time', 1)
