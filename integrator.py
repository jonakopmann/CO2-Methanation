from abc import abstractmethod

import casadi as ca
import numpy as np

from context import Context
from diffusion import Diffusion
from heat_conduction import HeatConduction
from parameters import Parameters
from plotter import Plotter
from reaction import Reaction


class Integrator:
    def __init__(self, params: Parameters, debug=False):
        self.params = params
        self.debug = debug

    @abstractmethod
    def get_x0(self):
        pass

    @abstractmethod
    def get_z0(self):
        pass

    @abstractmethod
    def get_alg_fl(self, ctx: Context):
        pass

    @abstractmethod
    def get_t(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def integrate(self):
        # create context
        ctx = Context(self.params)
        diff = Diffusion(self.params)
        reaction = Reaction(ctx)
        heat_cond = HeatConduction(ctx)

        # create ode variables
        ode_co2 = ca.SX(self.params.r_steps, 1)
        ode_ch4 = ca.SX(self.params.r_steps, 1)
        ode_h2o = ca.SX(self.params.r_steps, 1)
        ode_T = ca.SX(self.params.r_steps, 1)

        # make input dynamic
        alg_co2_fl, alg_h2o_fl, alg_ch4_fl, alg_T_fl = self.get_alg_fl(ctx)

        # create boundary conditions for the surface values
        alg_co2_surf = (ctx.w_co2_fl * ctx.rho_fl - ctx.D_co2_eff[-1] / ctx.beta_co2
                        * (ctx.w_co2_surf * ctx.rho_surf - ctx.w_co2[-1] * ctx.rho[-1]) / self.params.h
                        - ctx.w_co2_surf * ctx.rho_surf)
        alg_ch4_surf = (ctx.w_ch4_fl * ctx.rho_fl - ctx.D_ch4_eff[-1] / ctx.beta_ch4
                        * (ctx.w_ch4_surf * ctx.rho_surf - ctx.w_ch4[-1] * ctx.rho[-1]) / self.params.h
                        - ctx.w_ch4_surf * ctx.rho_surf)
        alg_h2o_surf = (ctx.w_h2o_fl * ctx.rho_fl - ctx.D_h2o_eff[-1] / ctx.beta_h2o
                        * (ctx.w_h2o_surf * ctx.rho_surf - ctx.w_h2o[-1] * ctx.rho[-1]) / self.params.h
                        - ctx.w_h2o_surf * ctx.rho_surf)
        alg_T_surf = (ctx.T_fl - self.params.lambda_eff / ctx.alpha * (ctx.T_surf - ctx.T[-1]) / self.params.h
                      - ctx.T_surf)

        # assign equations to ode for each radius i
        for i in range(self.params.r_steps):
            # get reaction rate
            r = reaction.get_r(i)

            # odes for w and T
            ode_co2[i] = (diff.get_term(ctx.w_co2, ctx.w_co2_surf, i, ctx.D_co2_eff[i])
                          + reaction.get_mass_term(self.params.M_co2, ctx.rho[i], self.params.v_co2, r))
            ode_ch4[i] = (diff.get_term(ctx.w_ch4, ctx.w_ch4_surf, i, ctx.D_ch4_eff[i])
                          + reaction.get_mass_term(self.params.M_ch4, ctx.rho[i], self.params.v_ch4, r))
            ode_h2o[i] = (diff.get_term(ctx.w_h2o, ctx.w_h2o_surf, i, ctx.D_h2o_eff[i])
                          + reaction.get_mass_term(self.params.M_h2o, ctx.rho[i], self.params.v_h2o, r))
            ode_T[i] = heat_cond.get_term(i) + reaction.get_heat_term(i, r)

            # # init binary diffusion coefficients
            # diff_i_eff.init(i)
            #
            # # alg for D_i_eff and p
            # alg_D_co2[i] = diff_i_eff.get_D_co2(i) - ctx.D_co2_eff[i]
            # alg_D_h2[i] = diff_i_eff.get_D_h2(i) - ctx.D_h2_eff[i]
            # alg_D_ch4[i] = diff_i_eff.get_D_ch4(i) - ctx.D_ch4_eff[i]
            #alg_p[i] = (self.params.M_0 * ctx.T[i]) / (ctx.M[i] * self.params.T_0) * self.params.p_0 - ctx.p[i]

        # create integrator
        dae = {
            'x': ca.veccat(ctx.w_co2, ctx.w_ch4, ctx.w_h2o, ctx.T),
            'z': ca.vertcat(ctx.w_co2_surf, ctx.w_ch4_surf, ctx.w_h2o_surf, ctx.T_surf,
                            ctx.w_co2_fl, ctx.w_ch4_fl, ctx.w_h2o_fl, ctx.T_fl),
            't': ctx.t,
            'ode': ca.vertcat(ode_co2, ode_ch4, ode_h2o, ode_T),
            'alg': ca.vertcat(alg_co2_surf, alg_ch4_surf, alg_h2o_surf, alg_T_surf,
                              alg_co2_fl, alg_ch4_fl, alg_h2o_fl, alg_T_fl)
        }

        options = {'regularity_check': True} # abstol
        if self.debug:
            options['verbose'] = True
            options['monitor'] = 'daeF'

        t0, t = self.get_t()
        integrator = ca.integrator('I', 'idas', dae, t0, t, options)

        # integrate
        return integrator(x0=self.get_x0(), z0=self.get_z0())

    def plot(self, res, folder):
        # create plotter and plot
        plotter = Plotter(self.params.t_i, np.linspace(0, self.params.r_max, self.params.r_steps + 1), res, self.params)

        path = f'{folder}/fw-{self.params.f_y}_deltaw-{self.params.delta_y}_fT-{self.params.f_T}_deltaT-{self.params.delta_T}_t-{self.params.t_max}'
        #plotter.plot_interval()
        #plotter.plot_w_i_feed_surface()
        #plotter.plot_conversion_per_temp()
        plotter.plot_closing_condition()
        #plotter.plot_dynamic_feed_w()
        #plotter.plot_dynamic_feed_T()
        #plotter.plot_frequency()
        #plotter.plot_cat_eff_thiele('Thiele eff')
        # if self.params.f_y == 1:
        #      plotter.plot_cat_eff()
        # else:
        #      plotter.plot_cat_eff_2(folder)
        #plotter.plot_frequency_center()

        #plotter.plot_3d_all()
        #plotter.plot_X_co2(folder)
        #plotter.plot_w(len(self.params.t_i)-1)
        #plotter.plot_feed_center('/home/jona/PycharmProjects/BachelorThesis/Figures/Plots/FeedCenter_3.pdf')
        #plotter.plot_feed_center_2()
        #plotter.plot_w_h2_fl_surf()
        #plotter.plot_T_fl_center()
        #plotter.plot_y_co2_fl_surf()
        #plotter.plot_y_h2_fl_surf()
        #plotter.plot_cat_eff()
        #plotter.plot_cat_eff_thiele('')
        #plotter.animate_w(os.path.join(path, 'weight.mp4'), 'Mass fractions over time', 1)
        #plotter.animate_T(os.path.join(path, 'temp.mp4'), 'Temperature over time', 1, )
        #plotter.animate_Y(os.path.join(path, 'yield.mp4'), 'Yield over time', 1)
        #plotter.animate_y(os.path.join(path, 'mole.mp4'), 'Mole fractions over time', 1)