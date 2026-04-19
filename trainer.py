import torch
import tqdm
import functools
import lpips
import os

from utils.perf_logger import perfLogger
from models.digitalFilm_v2 import digitalFilmv2
from tester import tester
from utils.dataset import filmDataset
from utils.amp_opt import AmpOptimizer
from utils.utils import filter_params
from utils import optimizer
from models.gan_loss import ganLoss
from utils.utils import set_random_seed


class trainer():
    def __init__(self, rank: int, world_size: int, global_config: dict, config: dict, train_dataset: filmDataset,
                 val_dataset: filmDataset, logger: perfLogger) -> None:
        self.rank = rank
        self.world_size = world_size
        self.logger: perfLogger = logger
        self.global_config = global_config
        self.config = config
        
        self.model_wo_ddp: digitalFilmv2 = digitalFilmv2(rank, global_config, config).to(rank)
        if self.global_config.checkpoint != "":
            checkpoint: dict = torch.load(self.global_config.checkpoint, map_location="cpu")
            self.model_wo_ddp.load_state_dict(checkpoint)
        
        if world_size > 1:
            self.model_ddp = torch.nn.parallel.DistributedDataParallel(self.model_wo_ddp, device_ids=[rank], find_unused_parameters=True)

        self.train_dataloader, val_dataloader, self.optimizers = self.__build_data_loader_optimizers(train_dataset, val_dataset, global_config)
        self.tester: tester = tester(global_config.metrics, val_dataloader, global_config, self.logger, rank)

        self.gan_loss: ganLoss = ganLoss(config.gan_loss, 1.0, 0.0).to(rank)
        self.lpips_loss: lpips.LPIPS = lpips.LPIPS(net='vgg').to(rank)

    def __build_data_loader_optimizers(self, train_dataset: filmDataset, val_dataset: filmDataset, opt: dict):
        set_random_seed(self.global_config.seed)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank, 
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=train_sampler,
            num_workers=opt.num_workers,
            pin_memory=True
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank, 
            shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            sampler=val_sampler,
            num_workers=opt.num_workers
        )

        optimizers = []
        for i, model_item in enumerate(self.model_wo_ddp._model):
            nowd_keys = {}
            ndim_dict = {name: para.ndim for name, para in model_item[1].named_parameters() if para.requires_grad}
            names, paras, para_groups = filter_params(model_item[1], ndim_dict, nowd_keys=nowd_keys)
            beta1, beta2 = map(float, opt.beta[i].split('_'))
            opt_clz = {
                'adam':  functools.partial(torch.optim.AdamW, betas=(beta1, beta2), fused=torch.cuda.is_available()),
                'adamw': functools.partial(torch.optim.AdamW, betas=(beta1, beta2), fused=torch.cuda.is_available()),
                'lamb':  functools.partial(optimizer.LAMBtimm, betas=(beta1, beta2), max_grad_norm=opt.grad_max_norm), # eps=1e-7
                'lion':  functools.partial(optimizer.Lion, betas=(beta1, beta2), max_grad_norm=opt.grad_max_norm),     # eps=1e-7
            }[opt.optimizer[i]]
            opt_kw = dict(lr=opt.lr[i], weight_decay=0)
            
            print(f'[vlip] optim={opt_clz}, opt_kw={opt_kw}\n')
            optimizers.append(AmpOptimizer(mixed_precision=self.global_config.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=model_item[0],
                                           paras=paras, grad_clip=self.global_config.grad_max_norm, n_gradient_accumulation=self.global_config.accum_steps))
            del names, paras, para_groups

        return train_loader, val_loader, optimizers

    def charbonnier_loss(self, x, y, eps=1e-3):
        return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))

    def gradient_loss(self, x, y):
        dx_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy_x = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx_y = y[:, :, :, 1:] - y[:, :, :, :-1]
        dy_y = y[:, :, 1:, :] - y[:, :, :-1, :]
        return (dx_x - dx_y).abs().mean() + (dy_x - dy_y).abs().mean()

    def color_stat_loss(self, fake, real):
        fake_mean = fake.mean(dim=[2,3])
        real_mean = real.mean(dim=[2,3])
        fake_std = fake.std(dim=[2,3])
        real_std = real.std(dim=[2,3])
        return (fake_mean - real_mean).abs().mean() + (fake_std - real_std).abs().mean()

    def tone_loss(self, fake, real):
        fake_y = 0.299 * fake[:,0] + 0.587 * fake[:,1] + 0.114 * fake[:,2]
        real_y = 0.299 * real[:,0] + 0.587 * real[:,1] + 0.114 * real[:,2]
        mean_loss = (fake_y.mean(dim=[1,2]) - real_y.mean(dim=[1,2])).abs().mean()
        std_loss = (fake_y.std(dim=[1,2]) - real_y.std(dim=[1,2])).abs().mean()
        return mean_loss + std_loss

    def __compute_G_loss(self, fake: torch.Tensor, real: dict):
        target = real["B"]

        pix_loss = self.charbonnier_loss(fake["out"], target)
        lpips_loss = self.lpips_loss(fake["out"], target).mean()
        grad_loss = self.gradient_loss(fake["out"], target)
        color_loss = self.color_stat_loss(fake["out"], target)
        tone_loss = self.tone_loss(fake["out"], target)
        lut_reg = (
            fake["tv3d"] * self.config.lambda_tv3d +
            fake["mn3d"] * self.config.lambda_mn3d +
            fake["tv4d"] * self.config.lambda_tv4d +
            fake["mn4d"] * self.config.lambda_mn4d
        )
        
        with torch.no_grad():
            logits_fake: torch.Tensor = self.model_ddp.module.d(fake["out"]).float()
        gan_loss: torch.Tensor = self.gan_loss(logits_fake, 1.0)

        g_loss = (
            pix_loss * self.config.lambda_pix +
            lpips_loss * self.config.lambda_lpips +
            grad_loss * self.config.lambda_grad +
            color_loss * self.config.lambda_color +
            tone_loss * self.config.lambda_tone +
            gan_loss * self.config.lambda_gan +
            lut_reg
        )

        return g_loss

    def __compute_D_loss(self, fake: torch.Tensor, real: dict):
        B, C, H, W = real["B"].shape 
        inp_rec_no_grad = torch.cat((real["B"], fake.detach()), dim=0) 
        logits: torch.Tensor = self.model_ddp.module.d(inp_rec_no_grad).float() 
        logits_real, logits_fake = logits[:B], logits[B:]
        d_loss: torch.Tensor = (self.gan_loss(logits_real, 1.0) +\
                                 self.gan_loss(logits_fake, 0.0)) * self.config.lambda_gan

        return d_loss

    def __train_one_step(self, epoch: int, step: int, item: dict):
        is_update_step: bool = (step % self.global_config.accum_steps == 0)
        amp_dtype = torch.float16 if self.global_config.fp16 == 1 else torch.bfloat16

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            fake = self.model_ddp.module.g(item["A"])
        
        g_loss = self.__compute_G_loss(fake, item)
        d_loss = self.__compute_D_loss(fake["out"], item)
        
        self.optimizers[0].backward_clip_step(False, g_loss)
        if is_update_step:
            self.optimizers[0].backward_clip_step(True, g_loss)

        self.optimizers[1].backward_clip_step(False, d_loss)
        if is_update_step:
            self.optimizers[1].backward_clip_step(True, d_loss)

        return g_loss, d_loss
        
    def train(self):
        # torch.distributed.barrier()
        
        for epoch in tqdm.tqdm(range(self.global_config.epoch), desc=f"{self.global_config.name} Training: "):
            self.model_ddp.train()
            for step, item in tqdm.tqdm(enumerate(self.train_dataloader), desc=f"Epoch {epoch + 1}", total=len(self.train_dataloader)):
                for(key, value) in zip(item.keys(), item.values()):
                    item[key] = value.to(self.rank)

                g_loss, d_loss = self.__train_one_step(epoch + 1, step + 1, item)
                global_step = self.global_config.save_epoch * len(self.train_dataloader) +\
                         (epoch + 1) * len(self.train_dataloader) + step
                self.logger.log_scalar("g_loss", g_loss, global_step)
                self.logger.log_scalar("d_loss", d_loss, global_step)
                self.logger.log_scalar("loss", g_loss + d_loss, global_step)

            tqdm.tqdm.write(f"Rank: {self.rank} - Epoch: {epoch + 1} done.")
            
            is_test_time: bool = ((epoch + 1) % self.global_config.test_per_epoch == 0)
            is_save_epoch: bool = ((epoch + 1) >= self.global_config.save_epoch or \
                            (epoch + 1) == (self.global_config.save_epoch // 2))                

            # torch.distributed.barrier()
            if is_test_time:
                self.model_ddp.eval()
                self.test(epoch)
                if is_save_epoch and self.rank == 0:
                    checkpoint_dir = f"{self.logger.get_log_dir()}/checkpoints/Epoch-{epoch + 1}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_path = f"{checkpoint_dir}/model.pth"
                    state_dict = self.model_wo_ddp.state_dict()
                    torch.save(state_dict, save_path)
                    self.logger.info(f"Save Model Weights to path: {save_path} .")
                # torch.distributed.barrier()

    def test(self, epoch: int = 0):
        with torch.no_grad():
            self.tester.test(self.model_ddp, epoch + 1)
