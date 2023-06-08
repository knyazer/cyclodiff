import torch
from abstractions import SamplingScheduler

class RhoWeightedSamplingScheduler(SamplingScheduler):
    def __init__(self, num_steps, rho):
        super().__init__(num_steps=num_steps, rho)

        assert rho > 0, "rho must be greater than 0, otherwise sigmas increase"
        self.rho = rho
        self.num_steps = num_steps

    def __call__(self, std_max=80, std_min=0.02):
        super().__call__(sigma_max, sigma_min)
        assert sigma_max > sigma_min, "sigma_max must be greater than sigma_min"
        assert sigma_min > 0, "sigma_min must be greater than 0"
        assert sigma_max > 0, "sigma_max must be greater than 0"

        step_indices = torch.arange(self.num_steps, dtype=torch.float32)
        stds = (std_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (std_min ** (1 / self.rho) - std_max ** (1 / self.rho))) ** self.rho
        stds = torch.cat([torch.as_tensor(stds), torch.zeros_like(stds[:1])]) # std of data point is zero
        
        # now, combine them into i, (current std, next std)
        return enumerate(zip(stds[:-1], stds[1:]))
        
class DefaultSamplingScheduler(SamplingScheduler):
    def __init__(self, num_steps):
        super().__init__(num_steps=num_steps)

        self.num_steps = num_steps

    def __call__(self):
        return range(self.num_steps, 0, -1)

