This repository contains the code for **"Assessing the Quality of Denoising Diffusion Models in Wasserstein Distance: Noisy Score and Optimal Bounds"** [<a href="https://arxiv.org/pdf/2506.09681">1</a>]. 

## Experiments
To run the inference please execute the below command with the parameters of your choice.

```bash
torchrun --standalone --nproc-per-node=8 src/models/time_aware_noisy_inf.py --run-path ./ --batch_size=4 --total_cycles_per_device=1 --std=2 --noise_dist="student_t" --dataset=church --timesteps 0 --root_folder_prefix src/data/temporal/noisy_score_images_test
```

FID can be comuted afterwords by providing the folder of generated images and the folder of ground truth images

```bash
python src/metrics/fid.py --force --noisy_folder="src/data/temporal/noisy_score_images_church/normal/0/" --real_images="src/data/lsun_church_10k.npz"
```

## Citations

```bibtex
@inproceedings{assessingqualitydenoisingdiffusion,
      title={Assessing the Quality of Denoising Diffusion Models in Wasserstein Distance: Noisy Score and Optimal Bounds}, 
      author={Vahan Arsenyan and Elen Vardanyan and Arnak Dalalyan},
      year={2025},
      journal={Advances in Neural Information Processing Systems, 2025}
      url={https://neurips.cc/virtual/2025/poster/119294}, 
}
```
