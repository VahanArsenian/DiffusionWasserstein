import torch 
from abc import ABC


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
        noise = torch.empty_like(like).normal_(generator=self.generator)
        return noise * self.std

class UniformNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.generator = generator

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(like).uniform_(-3**(0.5), to=3**(0.5), generator=self.generator)
        return noise * self.std

class LaplaceNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.laplace = torch.distributions.Laplace(loc=0, scale=std)

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        noise = self.laplace.sample(like.shape).to(like.device)
        return noise

class StudentTNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.std = std
        self.student_t = torch.distributions.StudentT(df=3, loc=0, scale=std)

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        noise = self.student_t.sample(like.shape).to(like.device)
        return noise

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
