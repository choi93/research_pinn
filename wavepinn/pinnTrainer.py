from omegaconf import DictConfig
import tensorflow as tf
from pathlib import Path
import time
import numpy as np
import logging
import yaml


from .model_pinn import WaveNet
from .utils.data_preproc import Wave2DDataGenerator, create_coordinate_grid
from .utils.module_io import createFolder, to_bin
from .losses import PhysicsLoss


log = logging.getLogger(__name__)

class PINNTrainer:
    def __init__(self, cfg: DictConfig, task='train', apply_cfg: str=None):
        """
        PINN 트레이너 초기화
        Args:
            cfg: 학습 설정
        """
        self.cfg = cfg
        self.task = task

        self.model = WaveNet(self.cfg.model)
        self.data_generator = Wave2DDataGenerator(self.cfg.data_config)


        if self.task == 'train':
            # Training state
            self.iter_count = 0
            self.loss_balance = tf.cast(self.cfg.training.physics_informed.loss_balance, tf.float32)
        
            # Setup logging
            log_dir = Path(cfg.training.logging.log_dir)
            checkpoint_dir = Path(cfg.training.checkpointing.save_dir)
            
            log_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # 모델, 데이터 생성기, 손실 함수 초기화
            self.data_generator = Wave2DDataGenerator(self.cfg.data_config)
            self.physics_loss = PhysicsLoss(wave_speed=self.cfg.training.wave_speed)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.training.learning_rate)
            
            # 로깅 설정
            self.setup_logging_files()
        elif self.task == 'apply':
            with open(apply_cfg, 'r') as f:
                self.apply_cfg = yaml.safe_load(f)

            domain_config = self.apply_cfg['domain']
            paths_config = self.apply_cfg['paths']  
            apply_config = self.apply_cfg['apply']

            self.nx = domain_config['nx']
            self.nz = domain_config['nz']
            self.nt = domain_config['nt']
            self.t_min = domain_config['t_min']
            self.t_max = domain_config['t_max']
            self.x_min = domain_config['x_min']
            self.x_max = domain_config['x_max']
            self.z_min = domain_config['z_min']
            self.z_max = domain_config['z_max']

            self.batch_size = apply_config['batch_size']
            self.debug = apply_config['debug']
            self.checkpoint_number = apply_config['checkpoint_number']

            self.trained_model_path = paths_config['trained_model']
            self.result_dir = paths_config['result_dir']

            createFolder(self.result_dir)

        
        
    def setup_logging_files(self):
        """로깅 파일 설정"""
        log_dir = self.cfg.training.logging.log_dir
        self.loss_file = open(f"{log_dir}/loss.txt", 'w')
        self.bc_loss_file = open(f"{log_dir}/loss_bc.txt", 'w')
        self.physics_loss_file = open(f"{log_dir}/loss_phys.txt", 'w')

    @tf.function
    def boundary_step(self, txz_boundary, u_boundary):
        """경계 조건 학습 스텝"""
        with tf.GradientTape() as tape:
            u_pred = self.model(txz_boundary)
            loss = tf.reduce_mean(tf.square(u_pred - u_boundary))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    @tf.function
    def pinn_step(self, txz_boundary, u_boundary, txz_collocation):
        """PINN 학습 스텝"""
        with tf.GradientTape(persistent=True) as tape:
            # 경계 조건 손실
            u_pred = self.model(txz_boundary)
            boundary_loss = tf.reduce_mean(tf.square(u_pred - u_boundary))
            
            # 물리 법칙 손실
            physics_loss_value = self.physics_loss(self.model, txz_collocation)
            
            # 전체 손실 - tf.cast로 타입 안정성 보장
            total_loss = physics_loss_value * self.loss_balance + boundary_loss
        
        # 개별적으로 그래디언트 계산
        grads = tape.gradient(total_loss, self.model.trainable_variables)

        # 그래디언트 적용
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return total_loss, physics_loss_value, boundary_loss

    def train(self):
        """전체 학습 과정"""
        t0 = time.time()
        
        # 경계 데이터 로드
        txz_boundary, u_boundary = self.data_generator.generate_boundary_data()
        
        # 경계 조건 학습
        for step in range(self.cfg.training.boundary_condition.steps):
            # 미니배치 샘플링
            idx = np.random.choice(len(txz_boundary), self.cfg.training.boundary_condition.sample_size)
            txz_batch = txz_boundary[idx]
            u_batch = u_boundary[idx]
            
            loss = self.boundary_step(txz_batch, u_batch)
            self.log_progress(loss, step, training_phase="boundary")

        # PINN 학습
        if self.cfg.training.physics_informed.enabled:
            dt = self.data_generator.t_max / (self.data_generator.n_snapshots - 1)
            nt = int(self.cfg.training.physics_informed.max_training_time / dt + 1)
            
            for time_step in range(nt):
                current_time = dt * (time_step + 1)
                
                for step in range(self.cfg.training.physics_informed.steps):
                    whole_step = time_step * self.cfg.training.physics_informed.steps + step + self.cfg.training.boundary_condition.steps
                    # 경계 데이터 샘플링
                    idx = np.random.choice(len(txz_boundary), self.cfg.training.physics_informed.sample_size)
                    txz_batch = txz_boundary[idx]
                    u_batch = u_boundary[idx]
                    
                    # 콜로케이션 포인트 생성
                    txz_collocation = self.data_generator.generate_collocation_points(
                        current_time, 
                        self.cfg.training.physics_informed.sample_size
                    )
                    
                    loss, physics_loss, boundary_loss = self.pinn_step(
                        txz_batch, u_batch, txz_collocation)
                    
                    self.log_progress(loss, whole_step, physics_loss, boundary_loss, 
                                    training_phase="pinn", time_step=time_step)

        training_time = time.time() - t0
        log.info(f'Training completed in {training_time:.2f} seconds')
        
        self.close_logging_files()

    def predict(self):
        """적용 과정"""
        model_path = self.trained_model_path
        checkpoint_number = self.checkpoint_number
        self.model.load_weights(f"{model_path}/wave2d_{checkpoint_number}")

        coords = create_coordinate_grid(
            self.nt, self.nx, self.nz,
            self.t_min, self.t_max,
            self.x_min, self.x_max,
            self.z_min, self.z_max
        )

        u_pred = self.model.predict(coords, batch_size=self.batch_size)
        u_pred = u_pred.reshape(self.nt, self.nx, self.nz)

        np.save(self.result_dir+'result.npy', u_pred)

        if self.debug:
            createFolder(self.result_dir+'debug/')
            for i in range(self.nt):
                to_bin(self.result_dir+'debug/'+str(i)+'.bin',u_pred[i])

        return u_pred

    def log_progress(self, loss, step, physics_loss=None, boundary_loss=None, 
                    training_phase="boundary", time_step=None):
        """학습 진행 상황 로깅"""
        if step % self.cfg.training.logging.save_frequency == 0:
            log_msg = f'[{training_phase}] Step {step}, Loss: {loss:.6e}'
            if physics_loss is not None:
                log_msg += f', Physics Loss: {physics_loss:.6e}, Boundary Loss: {boundary_loss:.6e}'
            if time_step is not None:
                log_msg += f', Time Step: {time_step}'
            log.info(log_msg)
            
            # 파일에 로깅
            self.loss_file.write(f"{loss:.6e}\n")
            if physics_loss is not None:
                self.physics_loss_file.write(f"{physics_loss:.6e}\n")
                self.bc_loss_file.write(f"{boundary_loss:.6e}\n")
            

        # 체크포인트 저장
        if step % self.cfg.training.logging.checkpoint_frequency == 0:
            self.save_checkpoint(step)

    def save_checkpoint(self, step):
        """체크포인트 저장"""
        checkpoint_path = f"{self.cfg.training.checkpointing.save_dir}/wave2d_{step}"
        self.model.save_weights(checkpoint_path)

    def close_logging_files(self):
        """로깅 파일들 닫기"""
        self.loss_file.close()
        self.bc_loss_file.close()
        self.physics_loss_file.close()


