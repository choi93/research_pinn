import os
from omegaconf import DictConfig
import hydra
from wavepinn.pinnTrainer import PINNTrainer

os.environ['CUDA_VISIBLE_DEVICES']='3'

@hydra.main(version_base="1.3", config_path="config", config_name="input_training")
def main(cfg: DictConfig):
    trainer = PINNTrainer(cfg, task='apply', apply_cfg='config/input_apply.yaml')
    trainer.predict()


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    main()


