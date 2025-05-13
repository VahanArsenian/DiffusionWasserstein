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
        self.generator = generator
        self.from_ = -3**(0.5) * std
        self.to_ = 3**(0.5) * std

    def __call__(self, like: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(like).uniform_(self.from_, to=self.to_, generator=self.generator)
        return noise 

class LaplaceNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
        self.scale = std / 2**(0.5)

    def __call__(self, like: torch.Tensor) -> torch.Tensor:

        noise = torch.as_tensor(cp.random.laplace(0, self.scale, like.shape, 
                                                  dtype=str(like.dtype).replace("torch.", "")))
        return noise

class StudentTNoise(Noise):
    def __init__(self, std: float = 1, generator=None):
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

# Unit test to check if the noise variance is 1
def test_noise_variance():
    std_to_test = 2
    noise = NoiseBuilder.build("normal", std=std_to_test)
    sample = torch.randn(3, 3, 32, 32)
    noisy_sample = noise(sample)
    assert abs(noisy_sample.std()-std_to_test) < 0.2, "Normal noise variance test failed"

    noise = NoiseBuilder.build("uniform", std=std_to_test)
    sample = torch.randn(3, 3, 32, 32)
    noisy_sample = noise(sample)
    assert abs(noisy_sample.std()-std_to_test) < 0.2, "Uniform noise variance check failed"

    noise = NoiseBuilder.build("laplace", std=std_to_test)
    sample = torch.randn(3, 3, 32, 32)
    noisy_sample = noise(sample)
    print(noisy_sample.std())
    assert abs(noisy_sample.std()-std_to_test) < 0.2, "Laplace noise variance check failed"

    noise = NoiseBuilder.build("student_t", std=std_to_test)
    sample = torch.randn(3, 3, 32, 32)
    noisy_sample = noise(sample)
    assert abs(noisy_sample.std()-std_to_test) < 0.2, "Student's t noise variance check failed"


if __name__ == "__main__":
    test_noise_variance()
