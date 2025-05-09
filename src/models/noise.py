import torch 
from abc import ABC


class Noise(ABC):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.generator = generator

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def add_score_noise(self, sample: torch.Tensor, current_beta_t) -> torch.Tensor:
        noise = self(sample)
        return sample + 2 * current_beta_t * noise

class NoNoise(Noise):
    def __init__(self, std: float = 0, generator=None):
        self.std = std
        self.generator = generator

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("No noise should not be called")
    
    def add_score_noise(self, sample: torch.Tensor, current_beta_t) -> torch.Tensor:
        return sample

class GaussianNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.generator = generator

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(like).normal_(generator=self.generator)
        return noise * self.std

class UniformNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.generator = generator

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(like).uniform_(-3**(0.5), to=3**(0.5), generator=self.generator)
        return noise * self.std


class NoiseBuilder:
        
    @classmethod
    def build(cls, distribution: str, std: float = 1, generator=None) -> Noise:
        if std == 0:
            return NoNoise(std=std, generator=generator)
        if distribution == "normal":
            return GaussianNoise(std=std, generator=generator)
        elif distribution == "uniform":
            return UniformNoise(std=std, generator=generator)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
