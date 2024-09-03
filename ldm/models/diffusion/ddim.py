"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

"""
DDIM采样器的实现: 这个采样器用于从扩散模型生成图像或其他数据表示。
时间步和参数的管理: 代码中设计了多种辅助函数来生成和管理采样过程中的参数，确保生成过程的稳定性和效率。
噪声预测与引导: 在采样过程中，使用噪声预测来调整潜在空间表示，通过指导调整生成结果以符合条件信息。
"""

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model   # model: 这是扩散模型的实例，通常是一个已训练的Latent Diffusion Model (LDM)。
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule   # schedule: 采样过程中使用的时间表类型，默认为 "linear"。

    def register_buffer(self, name, attr):  # register_buffer: 这是一个用于将变量（如张量）注册到类中的辅助方法。注册后的张量可以保存在 GPU 上以提高效率。
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    # make_schedule: 生成采样过程所需的时间步和参数。
    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,            # sample: 样本生成的主要方法。
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    # ddim_sampling: 实现了具体的DDIM采样过程。
    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        # 生成隐空间的初始向量，文生图。
        # 或者直接使用传入进来的x_T，图生图。
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        # 使用ddim的时间步
        # 这里内容看起来很多，但是其实很少，本质上就是取了self.ddim_timesteps，然后把它reversed一下
        if timesteps is None:
            timesteps   = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end  = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps   = self.ddim_timesteps[:subset_end]

        intermediates   = {'x_inter': [img], 'pred_x0': [img]}
        time_range      = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps     = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            # index是用来取得对应的调节参数的
            index   = total_steps - i - 1
            # 将步数拓展到bs维度
            ts      = torch.full((b,), step, device=device, dtype=torch.long)

            # --------------------------------------------------------------------------------- #
            #   替换特征的地方
            #   用于进行局部的重建，对部分区域的隐向量进行mask。
            #   对传入unet前的隐含层特征，我们利用mask将不重建的地方都替换成 原图加噪后的隐含层特征
            #   self.model.q_sample用于对输入图片进行ts步数的加噪
            # --------------------------------------------------------------------------------- #
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            # 进行采样
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            # 回调函数
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    # p_sample_ddim: 执行单个时间步的采样操作。
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        # 首先判断是否由neg prompt，unconditional_conditioning是由neg prompt获得的
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            # 一般都是有neg prompt的，所以进入到这里
            # 在这里我们对隐向量和步数进行复制，一个属于pos prompt，一个属于neg prompt
            # torch.cat默认堆叠维度为0，所以是在bs维度进行堆叠，二者不会互相影响
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            # 然后我们将pos prompt和neg prompt堆叠到一个batch中
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [
                            torch.cat([unconditional_conditioning[k][i], c[k][i]])
                            for i in range(len(c[k]))
                        ]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            # 堆叠完后，隐向量、步数和prompt条件一起传入网络中，将结果在bs维度进行使用chunk进行分割
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        # 一般不用
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # 根据采样器选择参数
        alphas      = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas      = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # 根据步数选择参数，
        # 这里的index就是上面循环中的total_steps - i - 1
        a_t         = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev      = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t     = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        # 公式中的最左边
        pred_x0             = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_  = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        # 公式的中间
        dir_xt              = (1. - a_prev - sigma_t**2).sqrt() * e_t
        # 公式最右边
        noise               = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise           = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev              = a_prev.sqrt() * pred_x0 + dir_xt + noise
        # 输出添加完公式的结果
        return x_prev, pred_x0

    # stochastic_encode: 对输入图像进行随机编码，用于在时间步数t上加入噪声。
    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    # decode: 使用DDIM采样从编码的潜在空间（latent space）中解码图像。
    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, mask, x0, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        # 使用ddim的时间步
        # 这里内容看起来很多，但是其实很少，本质上就是取了self.ddim_timesteps，然后把它reversed一下
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            
            # --------------------------------------------------------------------------------- #
            #   替换特征的地方
            #   用于进行局部的重建，对部分区域的隐向量进行mask。
            #   对传入unet前的隐含层特征，我们利用mask将不重建的地方都替换成 原图加噪后的隐含层特征
            #   self.model.q_sample用于对输入图片进行ts步数的加噪
            # --------------------------------------------------------------------------------- #
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                x_dec = img_orig * mask + (1. - mask) * x_dec

            # 进行单次采样
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
