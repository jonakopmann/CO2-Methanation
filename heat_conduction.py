from context import Context


class HeatConduction:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.params = ctx.params

    def get_lambda(self):
        # for now just return ref
        return self.params.lambda_eff

    def get_dr(self, r):
        if r == 0:
            # dy/dr(r=0)=0
            return 0
        return (self.ctx.T[r] - self.ctx.T[r - 1]) / self.params.h

    def get_dr2(self, r):
        if r == 0:
            # symmetrical
            return (self.ctx.T[r + 1] - 2 * self.ctx.T[r] + self.ctx.T[r + 1]) / (self.params.h ** 2)
        elif r == self.params.r_steps - 1:
            return (self.ctx.T[r - 1] - 2 * self.ctx.T[r] + self.ctx.T_surf) / (self.params.h ** 2)
        return (self.ctx.T[r - 1] - 2 * self.ctx.T[r] + self.ctx.T[r + 1]) / (self.params.h ** 2)

    def get_term(self, r):
        if r == 0:
            a = self.get_dr2(r)
        else:
            a = self.get_dr2(r) + (self.params.n / (r * self.params.h)) * self.get_dr(r)
        return 1e9 * self.get_lambda() / (self.params.roh_s * self.params.cp_s) * a
