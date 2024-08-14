import casadi as ca

from context import Context
from integrator import Integrator


class IntegratorSin(Integrator):
    def get_x0(self):
        return self.params.x0

    def get_z0(self):
        return self.params.z0

    def get_ode_fl(self, ctx: Context):
        alg_co2_fl = (self.params.w_co2_0 + self.params.delta_w * ca.sin(
            2 * ca.pi * self.params.f_w * ctx.t) - ctx.w_co2_fl)
        alg_h2_fl = (self.params.w_h2_0 - self.params.delta_w * ca.sin(
            2 * ca.pi * self.params.f_w * ctx.t) - ctx.w_h2_fl)
        alg_T_fl = self.params.T_0 + self.params.delta_T * ca.sin(2 * ca.pi * self.params.f_T * ctx.t) - ctx.T_fl
        alg_ch4_fl = (self.params.w_ch4_0 - ctx.w_ch4_fl)

        return alg_co2_fl, alg_h2_fl, alg_ch4_fl, alg_T_fl

    def get_t(self):
        return 0, self.params.t_i

    def run(self):
        # integrate
        res = self.integrate()
        self.plot(res, 'plots_sin')
