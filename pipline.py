import datetime
import torch
import os
import platform

from options.options import everyThingOptions
from utils.perf_logger import perfLogger
from trainer import trainer
from utils.dataset import filmDataset


def ddp_setup(rank: int, world_size: int, opt: everyThingOptions, logger: perfLogger):
    current_os = platform.system()
    if current_os == "Windows":
        logger.warning(f"[Rank {rank}] Detected Windows, disabling libuv...")
        os.environ["USE_LIBUV"] = "0"
        backend: str = "gloo"
    else:
        backend: str = "nccl"
        
    uri: str = f"tcp://{opt.opt.global_config.uri}:{opt.opt.global_config.port}"
    torch.distributed.init_process_group(backend=backend, 
                                         init_method=uri,
                                         rank=rank,
                                         world_size=world_size)
    torch.cuda.set_device(rank)

def worker(rank: int, world_size: int, options_path: str) -> tuple[everyThingOptions, perfLogger, trainer]:
    cur_options: everyThingOptions = everyThingOptions(options_path)
    cur_options.load_config()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger: perfLogger = perfLogger(f"{cur_options.opt.global_config.checkpoints_dir}/{timestamp}", f"{cur_options.opt.global_config.name}.log", cur_options.opt.global_config.log_mode) 
    logger.info(f"[{timestamp}] - Task: {cur_options.opt.global_config.name}.") 
    
    if rank == 0:
        os.makedirs(f"{logger.get_log_dir()}/checkpoints/", exist_ok=True)

    ddp_setup(rank, world_size, cur_options, logger)

    train_dataset: filmDataset = filmDataset(cur_options.opt.global_config, True) 
    val_dataset: filmDataset = filmDataset(cur_options.opt.global_config, False) 
    
    model_trainer: trainer = trainer(rank, world_size, cur_options.opt.global_config, cur_options.opt.model_config, 
                                     train_dataset, val_dataset, logger) 
    
    logger.info(f"Rank {rank}: Trainer creation complete.")

    run(rank, cur_options, logger, model_trainer)

    torch.distributed.barrier()
    if rank == 0:
        logger.info(f"All GPUs finished '{cur_options.opt.global_config.name}'. Proceeding...") 

    torch.distributed.destroy_process_group()
    
    return cur_options, logger, model_trainer

def run(rank: int, cur_options: everyThingOptions, logger: perfLogger, model_trainer: trainer) -> None:
    if cur_options.opt.global_config.is_train: 
        if rank == 0:
            logger.info("Starting training loop...")
        model_trainer.train()
    else:
        if rank == 0:
            logger.info("Starting evaluation...")
        model_trainer.test()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    options_path: str = "D:/projects/digitalFilm/options/digitalFilm.yaml"
    
    torch.multiprocessing.spawn( 
        worker,
        args=(world_size, options_path),
        nprocs=world_size,
        join=True
    )
