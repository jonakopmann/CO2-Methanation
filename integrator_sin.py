import casadi as ca

from context import Context
from integrator import Integrator


class IntegratorSin(Integrator):
    def get_x0(self):
        return self.params.x0

    def get_z0(self):
        return self.params.z0

    def get_alg_fl(self, ctx: Context):
        alg_co2_fl = (self.params.y_co2_0 + self.params.delta_y * ca.sin(
            2 * ca.pi * self.params.f_y * ctx.t) - ctx.y_co2_fl)
        alg_h2o_fl = (self.params.w_h2o_0 - ctx.w_h2o_fl)
        alg_T_fl = self.params.T_0 + self.params.delta_T * ca.sin(2 * ca.pi * self.params.f_T * ctx.t) - ctx.T_fl
        alg_ch4_fl = (self.params.w_ch4_0 - ctx.w_ch4_fl)

        return alg_co2_fl, alg_h2o_fl, alg_ch4_fl, alg_T_fl

    def get_t(self):
        return 0, self.params.t_i

    def run(self):
        # integrate
        res = self.integrate()
        self.plot(res, 'sin')