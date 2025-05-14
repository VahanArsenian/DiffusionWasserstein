from diffusers import DDPMPipeline, ImagePipelineOutput
from diffusers.pipelines.ddpm.pipeline_ddpm import XLA_AVAILABLE, is_torch_xla_available
import torch
import types
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, Union, List, Tuple
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

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


class QueuedDDPMPipeline(DDPMPipeline):

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample
            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

            if XLA_AVAILABLE:
                xm.mark_step()

            if t.item() in {250,500,750}:
                image_interim = (image / 2 + 0.5).clamp(0, 1)
                image_interim = image_interim.cpu().permute(0, 2, 3, 1).numpy()
                if output_type == "pil":
                    image_interim = self.numpy_to_pil(image_interim)

                self.queue.put((t, image_interim))



def create_patched_from_pretrained(model_name, noise, queue: Optional[torch.multiprocessing.Queue] = None):
    # Note: DDPMPipeline has output clamping, as it needs to output valid pixel values
    if queue is None:
        ddpm = DDPMPipeline.from_pretrained(model_name)
    else:
        ddpm = QueuedDDPMPipeline.from_pretrained(model_name)
        ddpm.queue = queue
    ddpm.scheduler.step = types.MethodType(step, ddpm.scheduler)
    ddpm.scheduler.noise_gen = noise
    return ddpm
