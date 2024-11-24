import numpy as np
import numpy.typing as npt
import yaml
from typing import Tuple
from module_io import from_bin

class Wave2DDataGenerator:
    """2D 파동 방정식 데이터를 생성하고 전처리하는 클래스
    
    이 클래스는 2D 파동 시뮬레이션 데이터의 로딩과 전처리를 담당하며,
    경계 조건과 콜로케이션 포인트 생성을 포함합니다.
    """
    
    def __init__(self, data_input_path: str = 'config/input_data.yaml'):
        """데이터 생성기를 설정 매개변수로 초기화
        
        Args:
            data_input_path: YAML 설정 파일 경로
        """
        # YAML 설정 파일 로드
        with open(data_input_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # 설정 매개변수 초기화
        snapshot_config = config['snapshot']
        domain_config = config['domain']
        paths_config = config['paths']
        
        # 스냅샷 관련 설정
        self.n_snapshots = snapshot_config['n_snap']  # 전체 스냅샷 수
        self.nx = snapshot_config['nx']  # x 방향 격자점 수
        self.nz = snapshot_config['nz']  # z 방향 격자점 수
        self.time_start = snapshot_config.get('time_start', 0)  # 시작 시간 인덱스
        
        # 도메인 관련 설정
        self.x_min = domain_config['x_min']  # x 좌표 최소값
        self.x_max = domain_config['x_max']  # x 좌표 최대값
        self.z_min = domain_config['z_min']  # z 좌표 최소값
        self.z_max = domain_config['z_max']  # z 좌표 최대값
        self.t_min = domain_config['t_min']  # 시작 시간
        self.t_max = domain_config['t_max']  # 종료 시간
        
        # 경로 설정
        self.snapshot_dir = paths_config['snapshot_dir']  # 스냅샷 파일 디렉토리
    
    def generate_boundary_data(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """경계 조건 데이터 생성
        
        좌표의 정규 격자를 생성하고 스냅샷 파일에서 해당하는 필드 값을 로드합니다.
        
        Returns:
            Tuple[NDArray, NDArray]: (좌표 데이터, 경계값 데이터)
            - 좌표 데이터: (전체_포인트, 3) 형태로 (t, x, z)
            - 경계값 데이터: (전체_포인트, 1) 형태로 필드값
        """
        coords = self.create_coordinate_grid(
            self.n_snapshots, self.nx, self.nz,
            self.t_min, self.t_max,
            self.x_min, self.x_max,
            self.z_min, self.z_max
        )
        values = self._load_snapshots(
            self.snapshot_dir, self.time_start,
            self.n_snapshots, self.nx, self.nz
        )
        return coords, values

    def generate_collocation_points(self, time: float, n_points: int) -> npt.NDArray:
        """특정 시간에 콜로케이션 포인트 생성
        
        주어진 시간에 도메인 내에서 무작위로 분포된 포인트를 생성합니다.
        
        Args:
            time: 콜로케이션 포인트를 생성할 시간
            n_points: 생성할 포인트 수
        
        Returns:
            NDArray: 생성된 콜로케이션 포인트 좌표, 형태 (n_points, 3)
        """
        # 각 차원에 대한 배열 생성
        t_col = np.full((n_points, 1), time)  # 모든 포인트에 대해 고정된 시간
        x_col = np.random.uniform(self.x_min, self.x_max, (n_points, 1))  # 무작위 x 좌표
        z_col = np.random.uniform(self.z_min, self.z_max, (n_points, 1))  # 무작위 z 좌표
        
        # 좌표를 단일 배열로 결합
        return np.concatenate([t_col, x_col, z_col], axis=1)

    def create_coordinate_grid(nt: int, nx: int, nz: int,
                              t_min: float, t_max: float,
                              x_min: float, x_max: float,
                              z_min: float, z_max: float) -> npt.NDArray:
        """메모리 효율적인 브로드캐스팅을 사용하여 정규 좌표 격자 생성
        
        Args:
            nt, nx, nz: 각 차원의 포인트 수
            t_min, t_max: 시간 도메인 경계
            x_min, x_max: X축 도메인 경계
            z_min, z_max: Z축 도메인 경계
            
        Returns:
            NDArray: 좌표 격자점, 형태 (nt*nx*nz, 3)
        """
        # 브로드캐스팅을 위한 적절한 차원의 1D 배열 생성
        t = np.linspace(t_min, t_max, nt)[:, None, None]  # 형태: (nt, 1, 1)
        x = np.linspace(x_min, x_max, nx)[None, :, None]  # 형태: (1, nx, 1)
        z = np.linspace(z_min, z_max, nz)[None, None, :]  # 형태: (1, 1, nz)
        
        # 브로드캐스팅을 사용하여 전체 좌표 배열 생성
        coords = np.broadcast_arrays(t, x, z)
        return np.column_stack([coord.reshape(-1) for coord in coords])

    def _load_snapshots(self, snapshot_dir: str, time_start: int,
                       n_snapshots: int, nx: int, nz: int) -> npt.NDArray:
        """바이너리 파일에서 스냅샷 데이터를 로드하고 변환
        
        Args:
            snapshot_dir: 스냅샷 파일이 있는 디렉토리
            time_start: 파일 이름 지정을 위한 시작 시간 인덱스
            n_snapshots: 로드할 총 스냅샷 수
            nx: x 방향 격자점 수
            nz: z 방향 격자점 수
            
        Returns:
            NDArray: 결합된 스냅샷 데이터, 형태 (전체_포인트, 1)
            
        Raises:
            Exception: 데이터 로딩 실패 시
        """
        # 성능 향상을 위한 리스트 사전 할당
        snapshots = [None] * n_snapshots
        
        try:
            for it in range(n_snapshots):
                # 0으로 채워진 인덱스로 파일 이름 생성
                filename = f"{snapshot_dir}{str(it+time_start).zfill(4)}"
                
                # 바이너리 데이터를 로드하고 열 벡터로 재구성
                snapshot = from_bin(filename, nx, nz)
                snapshots[it] = snapshot.reshape(-1, 1)
                
            # 모든 스냅샷을 수직으로 결합
            return np.concatenate(snapshots, axis=0)
            
        except Exception as e:
            # 오류가 발생한 스냅샷 식별
            error_index = next(i for i, s in enumerate(snapshots) if s is None)
            print(f"Error: Failed to load snapshot {error_index + time_start}")
            
            # 성공적으로 로드된 스냅샷 필터링
            loaded = [s for s in snapshots if s is not None]
            
            if loaded:
                # 데이터가 부분적으로 로드된 경우 경고 메시지 출력
                print(f"Warning: Partial data loaded ({len(loaded)}/{n_snapshots})")
                print(f"Successfully loaded snapshots: {time_start} ~ {time_start + len(loaded) - 1}")
                return np.concatenate(loaded, axis=0)
                
            # 데이터가 전혀 로드되지 않은 경우 예외 발생
            print(f"Error details: {str(e)}")
            raise Exception("Failed to load any snapshot data") from e

