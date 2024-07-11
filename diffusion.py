from parameters import Parameters


class Diffusion:
    def __init__(self, params: Parameters):
        self.params = params

    def get_D_i(self):
        # for now just return ref
        return self.params.D_i_eff

    def get_dr(self, y_i, r):
        if r == 0:
            # dy/dr(r=0)=0
            return 0
        return (y_i[r] - y_i[r - 1]) / self.params.h

    def get_dr2(self, y_i, y_i_surf, r):
        if r == 0:
            # symmetrical
            return (y_i[r + 1] - 2 * y_i[r] + y_i[r + 1]) / (self.params.h ** 2)
        elif r == self.params.r_steps - 1:
            return (y_i[r - 1] - 2 * y_i[r] + y_i_surf) / (self.params.h ** 2)
        return (y_i[r - 1] - 2 * y_i[r] + y_i[r + 1]) / (self.params.h ** 2)

    def calc(self, y_i, y_i_surf, r):
        if r == 0:
            return self.get_D_i() * self.get_dr2(y_i, y_i_surf, r)
        return self.get_D_i() * (
                self.get_dr2(y_i, y_i_surf, r) + (self.params.n / (r * self.params.h)) * self.get_dr(y_i, r))
