import hydra
from omegaconf import DictConfig
import logging
import os

from wavepinn.model_pinn import WaveNet
from wavepinn.utils.data_preproc import Wave2DDataGenerator
from wavepinn.loss.losses import PhysicsLoss


log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="config", config_name="input_training")
def main(cfg: DictConfig):
    # 설정 출력
    log.info(f"Training with config:\n{cfg}")
    
    # 모델, 데이터 생성기, 손실 함수 초기화
    model = WaveNet(cfg.model)
    data_generator = Wave2DDataGenerator('config/input_data.yaml')
    physics_loss = PhysicsLoss(wave_speed=cfg.training.wave_speed)
    
    # 트레이너 초기화 및 학습 실행
    trainer = PINNTrainer(model, data_generator, physics_loss, cfg)
    trainer.train()

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()