from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from tqdm import tqdm
# import sys
# import tqdm

import Code.Diffusion.Diffusion_Model as Diffusion_Model
from Code.Diffusion.Diffusion_Func import extract, default, linear_beta_schedule, cosine_beta_schedule, \
    sigmoid_beta_schedule, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, beta_setting

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


class GaussianDiffusion_E2ELoss(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_noise',
            beta_schedule='sigmoid',
            schedule_fn_kwargs=dict(),
            p2_loss_weight_gamma=0.,
            p2_loss_weight_k=1,
            ddim_sampling_eta=0.,
            auto_normalize=True,

    ):
        super().__init__()

        # model param init
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size
        self.objective = objective

        beta_schedule_fn = beta_setting(beta_schedule)
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance.clamp(min=1e-20))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** - p2_loss_weight_gamma)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_t, t, noise, x_start):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, guided=None, x_self_cond=None, clip_x_start=True):
        model_output = self.model(x, t, guided=guided, x_self_cond=x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, guided=None, x_self_cond=None):
        preds = self.model_predictions(x, t, guided=guided, x_self_cond=x_self_cond)
        x_start = preds.pred_x_start
        noise = preds.pred_noise
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_t=x, t=t, noise=noise,
                                                                                  x_start=x_start)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, guided=None, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, model_variance, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times,
                                                                                       guided=guided,
                                                                                       x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, raw_eeg, return_all_timesteps=False):
        # eeg ~ N(0, I)
        eeg = torch.randn_like(raw_eeg)
        denoise_way = [eeg]
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            eeg, x_start = self.p_sample(x=eeg, t=t, guided=raw_eeg, x_self_cond=self_cond)
            denoise_way.append(eeg)
        ret = eeg if not return_all_timesteps else torch.stack(denoise_way, dim=1)
        return ret

    @torch.no_grad()
    def ddim_sample(self, raw_eeg, return_all_timesteps=False):
        batch, device = raw_eeg.shape[0], raw_eeg.device
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        eeg = torch.randn(batch, 1000).to(device)
        denoise_way = [eeg]
        x_start = None

        for time, time_next in tqdm(time_pairs, leave=False):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(x=eeg, t=time_cond, guided=raw_eeg, x_self_cond=self_cond,
                                                             clip_x_start=True)

            if time_next < 0:
                eeg = x_start
                denoise_way.append(eeg)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # if time == 0:
            #     noise = 0
            # else:
            #     noise = torch.randn(batch, 1000).to(device)
            noise = 0

            eeg = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            denoise_way.append(eeg)

        ret = eeg if not return_all_timesteps else torch.stack(denoise_way, dim=1)

        return ret

    @torch.no_grad()
    def sample(self, raw_eeg, return_all_timesteps=False):
        if self.is_ddim_sampling:
            return self.ddim_sample(raw_eeg=raw_eeg, return_all_timesteps=return_all_timesteps)
        else:
            return self.p_sample_loop(raw_eeg=raw_eeg, return_all_timesteps=return_all_timesteps)

    def q_sample(self, x_start, t, noise=None):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, eeg_noise, eeg_pure, *args, **kwargs):
        b, l, device = eeg_pure.shape[0], eeg_pure.shape[1], eeg_pure.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = None
        noise = default(noise, lambda: torch.randn_like(eeg_pure))
        noise_eeg = self.q_sample(x_start=eeg_pure, t=t, noise=noise)
        # print(noise_eeg.shape, eeg_noise.shape)
        pre_noise = self.model(noise_eeg, time=t, guided=eeg_noise, x_self_cond=None)
        pre_x0 = self.predict_start_from_noise(x_t=noise_eeg, t=t, noise=pre_noise)
        clip = partial(torch.clamp, min=-1., max=1.)
        pre_x0 = clip(pre_x0)

        return noise, pre_noise, pre_x0

    def contrastive_forward(self, eeg_noise, eeg_pure, contrastive, *args, **kwargs):
        b, l, device = eeg_pure.shape[0], eeg_pure.shape[1], eeg_pure.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = None
        noise = default(noise, lambda: torch.randn_like(eeg_pure))
        noise_eeg = self.q_sample(x_start=eeg_pure, t=t, noise=noise)
        pre_noise = self.model(noise_eeg, time=t, guided=eeg_noise, semantic=contrastive, x_self_cond=None)
        # pre_x0 = self.predict_start_from_noise(x_t=noise_eeg, t=t, noise=pre_noise)
        # clip = partial(torch.clamp, min=-1., max=1.)
        # pre_x0 = clip(pre_x0)

        return noise, pre_noise
    
    def contrastive_forward_PN(self, eeg_noise, eeg_pure, contrastive, *args, **kwargs):
        b, l, device = eeg_pure.shape[0], eeg_pure.shape[1], eeg_pure.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = None
        noise = default(noise, lambda: torch.randn_like(eeg_pure))
        noise_eeg = self.q_sample(x_start=eeg_pure, t=t, noise=noise)
        pre_noise = self.model(noise_eeg, time=t, guided=eeg_noise, semantic=contrastive, x_self_cond=None)
        pre_x0 = self.predict_start_from_noise(x_t=noise_eeg, t=t, noise=pre_noise)
        clip = partial(torch.clamp, min=-1., max=1.)
        pre_x0 = clip(pre_x0)

        return noise, pre_noise, pre_x0

    @torch.no_grad()
    def contrastive_sample(self, raw_eeg, semantic, return_all_timesteps=False):
        if self.is_ddim_sampling:
            return self.contrastive_ddim_sample(raw_eeg=raw_eeg, semantic=semantic,
                                                return_all_timesteps=return_all_timesteps)
        else:
            return self.p_sample_loop(raw_eeg=raw_eeg, return_all_timesteps=return_all_timesteps)

    @torch.no_grad()
    def contrastive_ddim_sample(self, raw_eeg, semantic, return_all_timesteps=False):
        batch, device = raw_eeg.shape[0], raw_eeg.device
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # 9 8 . 8 7  7 6  6 5  54 43 32 21 10 0-1
        time_pairs = list(zip(times[:-1], times[1:]))

        eeg = torch.randn(batch, 1000).to(device)
        denoise_way = [eeg]
        x_start = None

        for time, time_next in tqdm(time_pairs, leave=False):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.contrastive_model_predictions(x=eeg, t=time_cond, guided=raw_eeg,
                                                                         semantic=semantic,
                                                                         x_self_cond=self_cond,
                                                                         clip_x_start=True)

            if time_next < 0:
                eeg = x_start
                denoise_way.append(eeg)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = 0

            eeg = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            denoise_way.append(eeg)

        ret = eeg if not return_all_timesteps else torch.stack(denoise_way, dim=1)

        return ret

    def contrastive_model_predictions(self, x, t, guided=None, semantic=None, x_self_cond=None, clip_x_start=True):
        model_output = self.model(x, t, guided=guided, semantic=semantic, x_self_cond=x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)
        return ModelPrediction(pred_noise, x_start)




if __name__ == '__main__':
    pass
