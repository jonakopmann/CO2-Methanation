import casadi as ca
import numpy as np

from context import Context
from integrator import Integrator
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


class IntegratorStep(Integrator):
    s: Slice = None
    res_final = {'xf': [], 'zf': []}

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

    def get_x0(self):
        if self.s.step == 0:
            return self.params.x0
        else:
            return self.res_final['xf'][:, -1]

    def get_z0(self):
        if self.s.step == 0:
            return self.params.z0
        else:
            return self.res_final['zf'][:, -1]

    def get_ode_fl(self, ctx: Context):
        if self.s.w:
            alg_co2_fl = (self.params.w_co2_0 + self.params.delta_w - ctx.w_co2_fl)
            alg_h2_fl = (self.params.w_h2_0 - self.params.delta_w - ctx.w_h2o_fl)
        else:
            alg_co2_fl = (self.params.w_co2_0 - self.params.delta_w - ctx.w_co2_fl)
            alg_h2_fl = (self.params.w_h2_0 + self.params.delta_w - ctx.w_h2o_fl)
        if self.s.T:
            alg_T_fl = self.params.T_0 + self.params.delta_T - ctx.T_fl
        else:
            alg_T_fl = self.params.T_0 - self.params.delta_T - ctx.T_fl
        alg_ch4_fl = (self.params.w_ch4_0 - ctx.w_ch4_fl)
        return alg_co2_fl, alg_h2_fl, alg_ch4_fl, alg_T_fl

    def get_t(self):
        t0 = float(self.params.t_i[self.s.step])
        t = self.params.t_i[self.s.step:self.s.step + self.s.step_size]
        return t0, t

    def run(self):
        slices = self.create_t_i()
        for s in slices:
            # set member variable so we can access slice in methods
            self.s = s

            # integrate
            res = self.integrate()
            skip = 1
            if s.step == 0:
                skip = 0

            self.res_final['xf'] = ca.horzcat(self.res_final['xf'], res['xf'][:, skip:])
            self.res_final['zf'] = ca.horzcat(self.res_final['zf'], res['zf'][:, skip:])

        self.plot(self.res_final, 'plots_step')
