from parameters import Parameters


class Diffusion:
    def __init__(self, params: Parameters):
        self.params = params

    def get_dr(self, w_i, r):
        if r == 0:
            # dy/dr(r=0)=0
            return 0
        return (w_i[r] - w_i[r - 1]) / self.params.h

    def get_dr2(self, w_i, w_i_surf, r):
        if r == 0:
            # symmetrical
            return (w_i[r + 1] - 2 * w_i[r] + w_i[r + 1]) / (self.params.h ** 2)
        elif r == self.params.r_steps - 1:
            return (w_i[r - 1] - 2 * w_i[r] + w_i_surf) / (self.params.h ** 2)
        return (w_i[r - 1] - 2 * w_i[r] + w_i[r + 1]) / (self.params.h ** 2)

    def get_term(self, w_i, w_i_surf, r, D_i_eff):
        if r == 0:
            a = self.get_dr2(w_i, w_i_surf, r)
        else:
            a = self.get_dr2(w_i, w_i_surf, r) + (self.params.n / (r * self.params.h)) * self.get_dr(w_i, r)
        return D_i_eff / self.params.epsilon * a
