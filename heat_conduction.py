from parameters import Parameters


class HeatConduction:
    def __init__(self, params: Parameters):
        self.params = params

    def get_lambda(self):
        # for now just return ref
        return self.params.lambda_eff

    def get_dr(self, T, r):
        if r == 0:
            # dy/dr(r=0)=0
            return 0
        return (T[r] - T[r - 1]) / self.params.h

    def get_dr2(self, T, T_surf, r):
        if r == 0:
            # symmetrical
            return (T[r + 1] - 2 * T[r] + T[r + 1]) / (self.params.h ** 2)
        elif r == self.params.r_steps - 1:
            return (T[r - 1] - 2 * T[r] + T_surf) / (self.params.h ** 2)
        return (T[r - 1] - 2 * T[r] + T[r + 1]) / (self.params.h ** 2)

    def calc(self, T, T_suf, r):
        if r == 0:
            return self.get_lambda() * self.get_dr2(T, T_suf, r)
        return self.get_lambda() * (
                self.get_dr2(T, T_suf, r) + (self.params.n / (r * self.params.h)) * self.get_dr(T, r))
