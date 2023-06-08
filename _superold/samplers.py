from abstractions import SamplerMethod

class EulerSampler(SamplerMethod):
    def __init__(self, denoiser):
        super().__init__(denoiser)

    def step(self, current_state, current_noise_std, next_noise_std, **kwargs):
        super().step(current_state, current_noise_std, next_noise_std, **kwargs)

        predicted_denoised = self.denoiser.step(current_state, current_noise_std, **kwargs)
        predicted_normalized_noise = (current_state - predicted_denoised) / current_noise_std
        next_state = current_state + (current_noise_std - next_noise_std) * predicted_normalized_noise

        return next_state
