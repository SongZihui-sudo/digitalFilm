import torch
import tqdm
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from models.base_model import model_base
from utils.perf_logger import perfLogger


class tester():
    def __init__(self, metrics: list, val_dataloader: torch.utils.data.DataLoader, global_config: dict, logger: perfLogger, rank: int) -> None:
        self.metrics: list = metrics
        self.logger = logger
        self.val_loader = val_dataloader
        self.global_config = global_config
        self.rank: int = rank
    
    @staticmethod
    def ssim(org_img: torch.Tensor, gen_img: torch.Tensor) -> float:
        ssim_compute = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(org_img.device)

        if gen_img.min() < 0:
            # [-1, 1] -> [0, 1]
            gen_img = gen_img.clamp(-1, 1) * 0.5 + 0.5
        else:
            # [0, 1]
            gen_img = gen_img.clamp(0, 1)

        if org_img.min() < 0:
            org_img = org_img.clamp(-1, 1) * 0.5 + 0.5
        else:
            org_img = org_img.clamp(0, 1)

        ssim = ssim_compute(org_img, gen_img)

        return float(ssim)

    @staticmethod
    def psnr(org_img: torch.Tensor, gen_img: torch.Tensor) -> float:
        psnr_compute = PeakSignalNoiseRatio((0, 1), dim=[0, 1, 2, 3]).to(org_img.device) 

        if gen_img.min() < 0:
            # [-1, 1] -> [0, 1]
            gen_img = gen_img.clamp(-1, 1) * 0.5 + 0.5
        else:
            # [0, 1]
            gen_img = gen_img.clamp(0, 1)

        if org_img.min() < 0:
            org_img = org_img.clamp(-1, 1) * 0.5 + 0.5
        else:
            org_img = org_img.clamp(0, 1)

        psnr = psnr_compute(org_img, gen_img)

        return float(psnr)

    def __test_one_step(self, model: model_base, item: dict, epoch: int, len_data: int, step: int):
        amp_dtype = torch.float16 if self.global_config.fp16 == 1 else torch.bfloat16 
        with torch.autocast(device_type=f"cuda:{self.rank}", dtype=amp_dtype, enabled=True, cache_enabled=True):
            for(key, value) in zip(item.keys(), item.values()):
                    item[key] = value.to(self.rank)
                    
            fake = model.module.g(item["A"])["out"]
            if "psnr" in self.metrics or "PSNR" in self.metrics:
                psnr: float = self.psnr(item["A"], fake)
                self.logger.log_scalar(f"{self.global_config.name}_test_psnr", psnr, epoch * len_data +\
                                        self.global_config.start_epoch * len_data)
            if "ssim" in self.metrics or "SSIM" in self.metrics:
                ssim: float = self.ssim(item["A"], fake)
                self.logger.log_scalar(f"{self.global_config.name}_test_ssim", ssim, epoch * len_data +\
                                        self.global_config.start_epoch * len_data)
            if not step:
                self.logger.log_image("original_DIGITAL", item["A"], epoch * len_data +\
                                    self.global_config.start_epoch * len_data, self.global_config.img_log_num_samples)
                self.logger.log_image("original_FILM", item["B"], epoch * len_data +\
                                    self.global_config.start_epoch * len_data, self.global_config.img_log_num_samples)
                self.logger.log_image("GEN", fake, epoch * len_data +\
                                    self.global_config.start_epoch * len_data, self.global_config.img_log_num_samples)
    
    def test(self, model: model_base, epoch: int = 0):
        self.logger.info(f"[Rank: {self.rank}] Starting Evaluation at Epoch {epoch}")

        for test_step, item in tqdm.tqdm(enumerate(self.val_loader), desc=f"Rank{self.rank}_{self.global_config.name}_Testing: ", total=len(self.val_loader)):
            self.__test_one_step(model, item, epoch, len(self.val_loader), test_step)
    