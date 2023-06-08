from abstractions import Denoiser

class NaiveDenoiser(Denoiser):
    def __init__(self, model):
        super().__init__(model)

    def apply(self, state, noise_std, **kwargs):
        super().apply(state, noise_std, **kwargs)
        # Naive formulation: just predict the normalized noise fraction of the image, so
        # D_theta(theta) = state + F_theta(state) * noise_std
        return state + self.model(state, noise_std, **kwargs) * noise_std
