import torch 
from abc import ABC
import cupy as cp


class Noise(ABC):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.generator = generator

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def add_score_noise(self, sample: torch.Tensor, current_beta_t) -> torch.Tensor:
        if self.std == 0:
            return sample
        noise = self(sample)
        return sample + 2 * current_beta_t * noise

class GaussianNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.generator = generator

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(like).normal_(std=self.std, generator=self.generator)
        return noise

class UniformNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.generator = generator
        self.from = -3**(0.5) * std
        self.to = 3**(0.5) * std

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(like).uniform_(self.from, to=self.to, generator=self.generator)
        return noise * self.std

# TODO: Move to cupy implementation, https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
class LaplaceNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.scale = std / 2**(0.5)

    def __call__(self, like: torch.Tensor) -> torch.Tensor:

        noise = torch.as_tensor(cp.random.laplace(0, self.scale, like.shape, 
                                                  dtype=str(like.dtype).replace("torch.", "")))
        # noise = self.laplace.sample(like.shape).to(like.device)
        return noise

# TODO: Same as above, move to cupy implementation
class StudentTNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.scale = std / 3**(0.5)

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        noise = torch.as_tensor(cp.random.standard_t(3, like.shape, 
                                                  dtype=str(like.dtype).replace("torch.", "")))

        return noise * self.scale

class NoiseBuilder:
        
    @classmethod
    def build(cls, distribution: str, std: float = 1, generator=None) -> Noise:
        if distribution == "normal":
            return GaussianNoise(std=std, generator=generator)
        elif distribution == "uniform":
            return UniformNoise(std=std, generator=generator)
        elif distribution == "laplace":
            return LaplaceNoise(std=std, generator=generator)
        elif distribution == "student_t":
            return StudentTNoise(std=std, generator=generator)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
