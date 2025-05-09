from diffusers import DDPMPipeline
import torch
import types
from diffusers.utils.torch_utils import randn_tensor
from typing import Union
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput

def step(
    self,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    generator=None,
    return_dict: bool = True,
):
    t = timestep

    prev_t = self.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    if self.config.prediction_type == "epsilon":
        noise_prediction = model_output 
        pred_prev_sample = (sample-current_beta_t/(beta_prod_t**0.5)*noise_prediction)/(current_alpha_t**0.5)

        pred_prev_sample = self.noise_gen.add_score_noise(pred_prev_sample, current_beta_t)
        # pred_prev_sample = pred_prev_sample + 2 * self.std * current_beta_t * torch.empty_like(sample).normal_(generator=generator)
    else:
        raise NotImplementedError(
            f"prediction_type given as {self.config.prediction_type} must be `epsilon` for the DDPMScheduler."
        )
    

    variance = 0
    if t > 0:
        device = model_output.device
        variance_noise = randn_tensor(
            model_output.shape, generator=generator, device=device, dtype=model_output.dtype
        )
        if self.variance_type == "fixed_small_log":
            variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
        elif self.variance_type == "learned_range":
            variance = self._get_variance(t, predicted_variance=predicted_variance)
            variance = torch.exp(0.5 * variance) * variance_noise
        else:
            variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

    pred_prev_sample = pred_prev_sample + variance

    if not return_dict:
        return (
            pred_prev_sample,
            None,
        )

    return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=None)


def create_patched_from_pretrained(model_name):
    ddpm = DDPMPipeline.from_pretrained(model_name)
    ddpm.scheduler.step = types.MethodType(step, ddpm.scheduler)
    return ddpm
