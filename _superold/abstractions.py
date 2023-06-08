
# Formulates what exactly the neural network should predict, so transforms the output of the model to the "predicted data_point", so that later we can apply any loss. Takes as an input "noised" state, and noise_std, both are parameters for the network, and outputs the predicted data point. Which means that the actual noise, that should be subtracted from state, has to be computed explicitly.
class Denoiser:
    def __init__(self, model):
        self.model = model

    def __call__(self, state, noise_std, **kwargs):
        assert isinstance(state, torch.Tensor), "state should be a tensor, I mean, obviously, no?"
        assert isinstance(noise_std, float), "noise_std should be a float, I am being strict, but it is the best.."
        assert state.dim() == 4, "state should be a 4D tensor, (batch_size, channels, width, height)"
        assert noise_std > 0, "noise_std should be positive"
        assert noise_std != 0, "noise_std could theoretically be zero, but it does not make sense to denoise data points, so, since we are being strict, just failing"
        assert not isinstance(self, Denoiser), "you should not call step from the instance of Denoiser, but from the instance of its child class, e.g. NaiveDenoiser"

# Formulates single step of sampling procedure, i.e. Euler, Heun or whatever
class SamplerMethod:
    def __init__(self, denoiser):
        self.denoiser = denoiser

    # current_state: the current, partially noised state. at 0 is the data point, at T is the "pure" noise
    def __call__(self, current_state, current_noise_std, next_noise_std, **kwargs):
        assert next_noise_std < current_noise_std, "next noise std should be less than current noise std, since we are going from pure noise (latents) to no noise (data points)"
        assert next_noise_std >= 0 and current_noise_std >= 0, "both noise stds should be more than 0"
        assert isinstance(current_state, torch.Tensor), "current_state should be a tensor, I mean, obviously, no?"
        assert len(current_state.shape) == 4, "current_state should be a 4D tensor, (batch_size, channels, width, height)"
        assert isinstance(current_noise_std, float) and isinstance(next_noise_std, float), "both noise stds must be floats; I am trying to be super strict here, so even if you provide '0' as std, it won't work, you should provide '0.0'"
        # current_noise std is not zero
        assert current_noise_std != 0, "current noise std should not be zero, since if its zero, the prediction of denoised image explodes. Look at the EulerSampler, for example. Again, being super strict here."
        assert current_noise_std != next_noise_std, "no point in making current_noise_std equal to the next_noise_std, since, well, the result will be the same as current state. I could have ignored this case, but strictness leads to better software."
        # fail if called from the instance of Sampler, and not child class
        assert not isinstance(self, SamplerMethod), "you should not call step from the instance of SamplerMethod, but from the instance of its child class, e.g. EulerSampler or HeunSampler"


# Sampling scheduler determines what is the "schedule" of noise standard deviations that we require. So, it should start from some large value, and finish at 0 (in fact, almost zero, but it is implementation dependendent).
class SamplingScheduler:
    def __init__(self, *args, **kwargs):
        # just don't init the SamplingScheduler by itself
        assert not isinstance(self, SamplingScheduler), "you should not call step from the instance of SamplingScheduler, but from the instance of its child class, e.g. RhoWeightedScheduler"

    # returns the generator of the schedule of noise stds
    def __call__(self, num_steps, *args, **kwargs):
        assert num_steps > 0, "num_steps should be positive"
        assert isinstance(num_steps, int), "num_steps should be an integer"

        assert not isinstance(self, SamplingScheduler), "you should not call step from the instance of SamplingScheduler, but from the instance of its child class, e.g. RhoWeightedScheduler"



