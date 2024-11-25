import os
import hydra
from omegaconf import DictConfig

from wavepinn.pinnTrainer import PINNTrainer

@hydra.main(version_base="1.3", config_path="config", config_name="input_training")
def main(cfg: DictConfig):
    trainer = PINNTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    main()
