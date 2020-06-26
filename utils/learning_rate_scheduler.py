import copy
import math


class TriangularSchedule:
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        inc_fraction: fraction of iterations spent in increasing stage (float)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction

    def __call__(self, iteration):
        if iteration <= self.cycle_length * self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle


class CosineAnnealingSchedule:
    def __init__(self, min_lr, max_lr, cycle_length):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length

    def __call__(self, iteration):
        if iteration <= self.cycle_length:
            unit_cycle = (1 + math.cos(iteration * math.pi / self.cycle_length)) / 2
            adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
            return adjusted_cycle
        else:
            return self.min_lr


class LinearWarmUp:
    def __init__(self, schedule, start_lr, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        start_lr: learning rate used at start of the warm-up (float)
        length: number of iterations used for the warm-up (int)
        """
        self.schedule = schedule
        self.start_lr = start_lr
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.finish_lr = copy.copy(schedule)(0)
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.length:
            return iteration * (self.finish_lr - self.start_lr) / (self.length) + self.start_lr
        else:
            return self.schedule(iteration - self.length)


class LinearCoolDown:
    def __init__(self, schedule, finish_lr, start_idx, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        finish_lr: learning rate used at end of the cool-down (float)
        start_idx: iteration to start the cool-down (int)
        length: number of iterations used for the cool-down (int)
        """
        self.schedule = schedule
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.start_lr = copy.copy(self.schedule)(start_idx)
        self.finish_lr = finish_lr
        self.start_idx = start_idx
        self.finish_idx = start_idx + length
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.start_idx:
            return self.schedule(iteration)
        elif iteration <= self.finish_idx:
            return (iteration - self.start_idx) * (self.finish_lr - self.start_lr) / (self.length) + self.start_lr
        else:
            return self.finish_lr


class OneCycleSchedule:
    def __init__(self, start_lr, max_lr, cycle_length, cooldown_length=0, finish_lr=None, inc_fraction=.5):
        """
        start_lr: lower bound for learning rate in triangular cycle (float)
        max_lr: upper bound for learning rate in triangular cycle (float)
        cycle_length: iterations between start and finish of triangular cycle: 2x 'stepsize' (int)
        cooldown_length: number of iterations used for the cool-down (int)
        finish_lr: learning rate used at end of the cool-down (float)
        """
        if (cooldown_length > 0) and (finish_lr is None):
            raise ValueError("Must specify finish_lr when using cooldown_length > 0.")
        if (cooldown_length == 0) and (finish_lr is not None):
            raise ValueError("Must specify cooldown_length > 0 when using finish_lr.")

        finish_lr = finish_lr if (cooldown_length > 0) else start_lr
        schedule = TriangularSchedule(min_lr=start_lr, max_lr=max_lr, cycle_length=cycle_length,
                                      inc_fraction=inc_fraction)  # inc_fraction=inc_fraction
        self.schedule = LinearCoolDown(schedule, finish_lr=finish_lr, start_idx=cycle_length, length=cooldown_length)

    def __call__(self, iteration):
        return self.schedule(iteration)


class CyclicalScheduleA:
    def __init__(self, schedule_class, cycle_length, cycle_length_decay=1.0, cycle_magnitude_decay=1.0,
                 stop_decay_iter=None, final_drop_iter=None, **kwargs):
        """
        schedule_class: class of schedule, expected to take `cycle_length` argument
        cycle_length: iterations used for initial cycle (int)
        cycle_length_decay: factor multiplied to cycle_length each cycle (float)
        cycle_magnitude_decay: factor multiplied learning rate magnitudes each cycle (float)
        kwargs: passed to the schedule_class
        """
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.stop_decay_iter = stop_decay_iter
        self.cycle_idx_stop = None
        self.final_drop_iter = final_drop_iter
        self.kwargs = kwargs
        self.current_cycle_idx = 0
        self.base_lr = 0

    def __call__(self, iteration):
        cycle_idx = 0
        cycle_length = self.length
        idx = self.length
        while idx <= iteration:
            cycle_length = math.ceil(cycle_length * self.length_decay)
            cycle_idx += 1
            idx += cycle_length

        if cycle_idx > self.current_cycle_idx:
            self.kwargs['max_lr'] = self.kwargs['max_lr'] * self.magnitude_decay
            self.current_cycle_idx = cycle_idx

        cycle_offset = iteration - idx + cycle_length
        # print(self.current_cycle_idx, self.kwargs['max_lr'])

        schedule = self.schedule_class(cycle_length=cycle_length, **self.kwargs)
        if self.stop_decay_iter and (self.stop_decay_iter == iteration):
            self.cycle_idx_stop = cycle_idx
            # self.length_decay = 1
        cycle_idx = self.cycle_idx_stop if self.cycle_idx_stop else cycle_idx

        if self.final_drop_iter and iteration >= self.final_drop_iter:
            return 1e-6
        current_lr = schedule(cycle_offset)
        return current_lr if current_lr > self.kwargs['min_lr'] else self.kwargs['min_lr']
        # * self.magnitude_decay ** cycle_idx


class CyclicalSchedule:
    def __init__(self, schedule_class, cycle_length, cycle_length_decay=1.0, cycle_magnitude_decay=1.0,
                 stop_decay_iter=None, final_drop_iter=None, **kwargs):
        """
        schedule_class: class of schedule, expected to take `cycle_length` argument
        cycle_length: iterations used for initial cycle (int)
        cycle_length_decay: factor multiplied to cycle_length each cycle (float)
        cycle_magnitude_decay: factor multiplied learning rate magnitudes each cycle (float)
        kwargs: passed to the schedule_class
        """
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.stop_decay_iter = stop_decay_iter
        self.cycle_idx_stop = None
        self.final_drop_iter = final_drop_iter
        self.kwargs = kwargs
        self.base_lr = 0

    def __call__(self, iteration):
        cycle_idx = 0
        cycle_length = self.length
        idx = self.length
        while idx <= iteration:
            cycle_length = math.ceil(cycle_length * self.length_decay)
            cycle_idx += 1
            idx += cycle_length

        cycle_offset = iteration - idx + cycle_length

        schedule = self.schedule_class(cycle_length=cycle_length, **self.kwargs)
        if self.stop_decay_iter and (self.stop_decay_iter == iteration):
            self.cycle_idx_stop = cycle_idx
            # self.length_decay = 1
        cycle_idx = self.cycle_idx_stop if self.cycle_idx_stop else cycle_idx

        if self.final_drop_iter and iteration >= self.final_drop_iter:
            return 1e-6
        return schedule(cycle_offset) * self.magnitude_decay ** cycle_idx


if __name__ == '__main__':
    cycle_length = 100
    total_iter = 2008
    stop_decay_iter = None
    final_drop_iter = None
    lrs = []
    lr_schedule = CyclicalSchedule(TriangularSchedule, cycle_length, min_lr=1e-4, max_lr=1e-3, inc_fraction=.9,
                                   cycle_length_decay=.92,
                                   cycle_magnitude_decay=.96, stop_decay_iter=stop_decay_iter,
                                   final_drop_iter=final_drop_iter)
    # lr_schedule = LinearWarmUp(
    #     OneCycleSchedule(start_lr=5e-4, max_lr=3e-2, cycle_length=1200, cooldown_length=300, finish_lr=1e-7, inc_fraction=.2),
    #     start_lr=1e-5,
    #     length=200,
    # )
    for i in range(total_iter):
        lrs.append(lr_schedule(iteration=i))

    import pylab as plt

    plt.plot(lrs)
    plt.show()
