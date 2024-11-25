import tensorflow as tf

class PhysicsLoss:
    """물리 기반 손실 함수를 처리하는 클래스"""
    
    def __init__(self, wave_speed):
        self.v = tf.constant(wave_speed, dtype=tf.float32)
    
    @tf.function
    def compute_wave_residual(self, model, txz):
        """파동 방정식 잔차 계산"""
        txz = tf.cast(txz, tf.float32)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(txz)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(txz)
                u = model(txz)
            
            grads = tape1.gradient(u, txz)
            u_t, u_x, u_z = tf.split(grads, 3, axis=1)
        
        u_tt = tape2.gradient(u_t, txz)[:, 0:1]
        u_xx = tape2.gradient(u_x, txz)[:, 1:2]
        u_zz = tape2.gradient(u_z, txz)[:, 2:3]
        
        return self.v**2 * (u_xx + u_zz) - u_tt

    def __call__(self, model, txz):
        residual = self.compute_wave_residual(model, txz)
        return tf.reduce_mean(tf.square(residual))
