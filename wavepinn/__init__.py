from .pinnTrainer import PINNTrainer
from .model_pinn import WaveNet
from .utils.data_preproc import Wave2DDataGenerator
from .losses import PhysicsLoss

__version__ = '0.1.0'

__all__ = [
    'PINNTrainer',
    'WaveNet',
    'Wave2DDataGenerator',
    'PhysicsLoss',
    '__version__'
] 