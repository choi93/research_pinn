import tensorflow as tf
from tf_siren import SinusodialRepresentationDense 

class WaveNet(tf.keras.Model):
    """
    SIREN (Sinusoidal Representation Networks) 또는 일반 신경망을 구현하는 모델 클래스
    입력 신호를 파동 함수나 일반 활성화 함수를 통해 처리하여 출력을 생성
    """
    
    def __init__(self, config):
        """
        모델 초기화
        Args:
            config: 모델 구성에 필요한 하이퍼파라미터들을 포함하는 설정 객체
                   (input_dim, output_dim, n_layers, n_units, activation 등)
        """
        super().__init__()
        self.config = config
        # 모델의 전체 레이어 구조를 생성하고 Sequential 모델로 구성
        self.model = self._build_layers()
    
    def _build_layers(self):
        """
        모델의 레이어 구조를 생성하는 메서드
        Returns:
            tf.keras.Sequential: 구성된 레이어들을 포함하는 Sequential 모델
        """
        # 입력 레이어 정의 - 입력 데이터의 형태를 지정
        layers = [tf.keras.layers.InputLayer(input_shape=(self.config.input_dim,))]
        
        # 히든 레이어들을 생성하고 리스트에 추가
        # 각 레이어는 동일한 유닛 수(n_units)를 가지며, 지정된 활성화 함수를 사용
        if self.config.activation == 'sine':
            layers.extend([
                SinusodialRepresentationDense(
                    self.config.n_units,
                    activation=self.config.activation,
                    w0=1.0,
                    name=f'siren_layer_{i}'
                ) for i in range(self.config.n_layers)
            ])
        else:
            layers.extend([
                tf.keras.layers.Dense(
                    self.config.n_units,
                    activation=self.config.activation,
                    name=f'dense_layer_{i}'
                ) for i in range(self.config.n_layers)
            ])


        # 출력 레이어 추가
        # 출력 차원과 최종 활성화 함수를 설정하여 원하는 출력 형태 생성
        layers.append(tf.keras.layers.Dense(
            self.config.output_dim,
            activation=self.config.output_activation
        ))
        
        # 모든 레이어를 Sequential 모델로 구성하여 반환
        return tf.keras.Sequential(layers)
    
    @tf.function
    def call(self, inputs):
        """
        모델의 순전파(forward pass)를 수행하는 메서드
        XLA 컴파일러를 통해 최적화된 실행을 수행

        Args:
            inputs: 모델에 입력되는 텐서
        Returns:
            모델의 출력 텐서
        """
        return self.model(inputs) 