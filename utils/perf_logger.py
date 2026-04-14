import os
import logging
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import yaml
import torchvision


class perfLogger:
    def __init__(self, log_dir: str = "logs", log_filename: str = "train.log", mode: str = "tensorboard", rank: int = 0) -> None:
        '''
            Logger + TensorBoard Wrapper

            Args:

            log_dir (str): Log folder path

            log_filename (str): Log file name

            mode (str): "tensorboard" | "none"

            rank (int): Distributed rank, only writes to TensorBoard when rank=0
        '''
        os.makedirs(log_dir, exist_ok=True)

        # 设置 logging
        self.logger: logging.Logger = logging.getLogger(f"TrainingLogger-{rank}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # 文件 handler
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # 终端 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # 避免重复添加
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        # TensorBoard
        self.tb_writer: SummaryWriter | None = None
        if mode == "tensorboard" and rank == 0:
            self.tb_writer = SummaryWriter(log_dir=log_dir)

        self.__log_dir: str = log_dir

    def get_log_dir(self) -> str:
        return self.__log_dir

    # logging 基本方法
    def info(self, message: str) -> None: 
        self.logger.info(message)
        
    def warning(self, message: str) -> None: 
        self.logger.warning(message)
        
    def error(self, message: str) -> None: 
        self.logger.error(message)
        
    def debug(self, message: str) -> None: 
        self.logger.debug(message)

    # TensorBoard scalar
    def log_scalar(self, tag: str, value, step: int) -> None:
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
            self.tb_writer.flush()

    # TensorBoard image
    def log_image(self, tag: str, fig: torch.Tensor, step: int, num_samples: int = 4) -> None:
        total: int = fig.size(0)
        num_samples = min(num_samples, total)
        indices = list(range(min(num_samples, total)))
        selected_imgs = fig[indices]
        grid: torch.Tensor = torchvision.utils.make_grid(selected_imgs, nrow=2, normalize=True, scale_each=True)
        
        if self.tb_writer:
            assert grid.dim() in (2, 3), f"log_image expects CHW or HW tensor, got shape {grid.shape}"
            if grid.dim() == 2:  # HW -> 1HW
                grid = grid.unsqueeze(0)
            self.tb_writer.add_image(tag, grid, step, dataformats="CHW")
            self.tb_writer.flush()

    # 关闭
    def close(self) -> None:
        self.info("Closing logger and TensorBoard writer.")
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None
            