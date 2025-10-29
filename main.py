import os
import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
from visualizer import MapVisualizer
from collections import deque
from cal_pose import PoseCalculator, GlobalMapManager, normalize_angle
import random


def generate_random_pairs(buffer_size, num_pairs=None):
    """
    각 프레임에 대해 1번째 또는 2번째 이전 버퍼와 랜덤하게 비교
    예: 5번 버퍼는 4번 또는 3번과 비교
    """
    if num_pairs is None:
        num_pairs = buffer_size // 2
    
    pairs = []
    
    # 각 프레임마다 1개 또는 2개 이전 버퍼와 비교
    for i in range(buffer_size):
        # 비교 가능한 이전 버퍼 목록
        possible_prev = []
        if i >= 1:
            possible_prev.append(i - 1)  # 1번째 이전
        if i >= 2:
            possible_prev.append(i - 2)  # 2번째 이전
        
        # 랜덤하게 선택
        if len(possible_prev) > 0:
            prev_idx = random.choice(possible_prev)
            pairs.append((prev_idx, i))
    
    # num_pairs만큼만 반환 (중복 제거 후)
    pairs = list(set(pairs))  # 중복 제거
    random.shuffle(pairs)
    
    return pairs[:num_pairs]


class FarmMapper:
    def __init__(self, dataset_root, project_root, lidar_resolution=5, depth_scale=1000, 
                 buffer_size=10, frame_interval=1, num_random_pairs=None):
        self.dataset_root = Path(dataset_root)
        self.project_root = Path(project_root)
        self.lidar_resolution = lidar_resolution
        self.depth_scale = depth_scale
        self.buffer_size = buffer_size
        self.frame_interval = frame_interval
        self.num_random_pairs = num_random_pairs if num_random_pairs is not None else buffer_size // 2
        
        self.frame_dir = self.dataset_root / "frame_R"
        self.depth_dir = self.dataset_root / "depth_png_R"
        self.lidar_dir = self.dataset_root / "lidar" / "fov360"
        self.vis_dir = self.project_root / "visualization"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = MapVisualizer(self.vis_dir)
        
        self.robot_trajectory = [np.array([0.0, 0.0], dtype=np.float32)]
        self.robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_yaw = 0.0
        
        self.point_cloud_buffer = deque(maxlen=buffer_size)
        self.frame_id_buffer = deque(maxlen=buffer_size)
        self.local_occupancy_buffer = deque(maxlen=buffer_size)
        
        self.buffer_positions = deque(maxlen=buffer_size)
        self.buffer_yaws = deque(maxlen=buffer_size)
        
        self.all_frame_ids = []
        
        self.camera_intrinsics = self.estimate_camera_intrinsics()
        self.pose_calculator = PoseCalculator()
        self.map_manager = GlobalMapManager()
        
        print(f"FarmMapper initialized")
        print(f"LiDAR resolution: {self.lidar_resolution}cm/cell")
        print(f"Window size: {self.buffer_size} frames")
        print(f"Frame interval: {self.frame_interval} frames")
        print(f"Random pairs per window: {self.num_random_pairs}")
    
    def estimate_camera_intrinsics(self):
        fx = 1258.97
        fy = 1258.97
        cx = 916.48
        cy = 553.83
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    def get_file_list(self):
        files = sorted(self.frame_dir.glob("R_*.png"))
        return [f.stem.replace("R_", "") for f in files]
    
    def load_frame(self, frame_id):
        frame_path = self.frame_dir / f"R_{frame_id}.png"
        if frame_path.exists():
            return cv2.imread(str(frame_path))
        return None
    
    def load_depth(self, frame_id):
        depth_path = self.depth_dir / f"{frame_id}.png"
        if depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            return depth.astype(np.float32) / self.depth_scale
        return None
    
    def load_lidar(self, frame_id, min_distance=2, max_distance=30.0):
        lidar_path = self.lidar_dir / f"{frame_id}.ply"
        if lidar_path.exists():
            pcd = o3d.io.read_point_cloud(str(lidar_path))
            points = np.asarray(pcd.points)
            points = points / 100.0
            transformed = np.zeros_like(points)
            transformed[:, 0] = -points[:, 1]
            transformed[:, 1] = points[:, 0]
            transformed[:, 2] = points[:, 2]
            
            distances = np.sqrt(transformed[:, 0]**2 + transformed[:, 1]**2)
            valid_points = transformed[(distances >= min_distance) & 
                                   (distances <= max_distance)] 
                                #    (transformed[:, 2] >= 0.01)]
            return valid_points
        return None
    
    def depth_to_pointcloud(self, depth, K):
        h, w = depth.shape
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        mask = depth > 0
        
        X = (xx[mask] - K[0, 2]) * depth[mask] / K[0, 0]
        Y = (yy[mask] - K[1, 2]) * depth[mask] / K[1, 1]
        Z = depth[mask]
        
        points = np.stack([X, Y, Z], axis=1)
        return points
    
    def merge_point_clouds(self, lidar_points, depth_points):
        if lidar_points is None:
            return depth_points
        if depth_points is None:
            return lidar_points
        
        lidar_down = self.pose_calculator.downsample_points(lidar_points)
        depth_down = self.pose_calculator.downsample_points(depth_points)
        
        merged = np.vstack([lidar_down, depth_down])
        
        return merged
    
    def points_to_ground_occupancy(self, points, height_threshold=0.05):
        if len(points) == 0:
            return {}
        
        ground_points = points[points[:, 2] < height_threshold]
        obstacle_points = points[points[:, 2] >= height_threshold]
        
        occupancy = {}
        
        for p in ground_points:
            gx = round(p[0], 2)
            gy = round(p[1], 2)
            key = (gx, gy)
            occupancy[key] = 0
        
        for p in obstacle_points:
            gx = round(p[0], 2)
            gy = round(p[1], 2)
            key = (gx, gy)
            occupancy[key] = 1
        
        return occupancy
    
    def process_frame(self, frame_id):
        print(f"Processing Frame {frame_id}")
        
        rgb = self.load_frame(frame_id)
        depth = self.load_depth(frame_id)
        lidar = self.load_lidar(frame_id)
        
        if lidar is None and depth is None:
            print("  No data available")
            return False
        
        depth_points = None
        if depth is not None:
            depth_points = self.depth_to_pointcloud(depth, self.camera_intrinsics)
        
        curr_points = self.merge_point_clouds(lidar, depth_points)
        
        if len(curr_points) < 100:
            print("  Too few points")
            return False
        
        local_occupancy = self.points_to_ground_occupancy(curr_points)
        
        self.point_cloud_buffer.append(curr_points.copy())
        self.frame_id_buffer.append(frame_id)
        self.local_occupancy_buffer.append(local_occupancy)
        
        if len(self.buffer_positions) == 0:
            self.buffer_positions.append(np.array([0.0, 0.0]))
            self.buffer_yaws.append(0.0)
        
        random_pairs = generate_random_pairs(len(self.point_cloud_buffer), self.num_random_pairs)
        print(f"  Random pairs: {random_pairs}")
        
        result = self.pose_calculator.estimate_motion_multiframe(
            self.point_cloud_buffer, 
            refine_rotation=True,
            random_pairs=random_pairs,
            current_global_pos=self.robot_pos,
            current_global_yaw=self.robot_yaw
        )
        
        if result is not None:
            if len(result) == 4:
                frame_positions, frame_yaws, global_positions, global_yaws = result
            else:
                frame_positions, frame_yaws = result
                global_positions, global_yaws = None, None
            
            positions_changed = []
            yaws_changed = []
            
            for i in range(len(self.buffer_positions)):
                old_pos = self.buffer_positions[i]
                new_pos = frame_positions[i]
                old_yaw = self.buffer_yaws[i]
                new_yaw = frame_yaws[i]
                
                pos_diff = np.linalg.norm(old_pos - new_pos)
                yaw_diff = abs(normalize_angle(new_yaw - old_yaw))
                
                # 5cm 또는 2도 이상 변화 시 업데이트
                if pos_diff > 0.05 or yaw_diff > np.deg2rad(2):
                    positions_changed.append(i)
                    print(f"    Frame {i} updated: "
                          f"pos ({old_pos[0]:.2f}, {old_pos[1]:.2f}) -> ({new_pos[0]:.2f}, {new_pos[1]:.2f}), "
                          f"yaw {np.degrees(old_yaw):.1f}° -> {np.degrees(new_yaw):.1f}°")
            
            if len(positions_changed) > 0:
                print(f"  {len(positions_changed)} frame(s) updated, rebuilding global map...")
                
                # 전역 맵 재구축
                self.map_manager.update_global_map_for_buffer(
                    frame_positions, frame_yaws, 
                    self.buffer_positions, self.buffer_yaws,
                    self.local_occupancy_buffer, self.robot_pos, self.robot_yaw
                )
                
                # 버퍼 내 모든 프레임의 상대 위치 업데이트
                for i in range(len(frame_positions)):
                    if i < len(self.buffer_positions):
                        self.buffer_positions[i] = frame_positions[i].copy()
                        self.buffer_yaws[i] = frame_yaws[i]
                    else:
                        self.buffer_positions.append(frame_positions[i].copy())
                        self.buffer_yaws.append(frame_yaws[i])
                
                # robot_pos는 그대로 유지 (버퍼의 첫 프레임 위치)
                # 단, 궤적의 마지막 점은 현재 계산된 전역 위치로 업데이트
                if global_positions is not None and len(self.robot_trajectory) > 0:
                    self.robot_trajectory[-1] = global_positions[0].copy()
                    print(f"  Updated trajectory point: ({global_positions[0][0]:.2f}, {global_positions[0][1]:.2f})")
            else:
                # 변화 없음: 새 프레임만 추가
                self.buffer_positions.append(frame_positions[-1].copy())
                self.buffer_yaws.append(frame_yaws[-1])
                
                # 새 프레임의 occupancy를 전역 맵에 추가
                if global_positions is not None and global_yaws is not None:
                    new_global_pos = global_positions[-1]
                    new_global_yaw = global_yaws[-1]
                else:
                    # global_positions가 없으면 수동 계산
                    rel_pos = frame_positions[-1]
                    rel_yaw = frame_yaws[-1]
                    
                    cos_yaw = np.cos(self.robot_yaw)
                    sin_yaw = np.sin(self.robot_yaw)
                    
                    new_global_pos = self.robot_pos.copy()
                    new_global_pos[0] += cos_yaw * rel_pos[0] - sin_yaw * rel_pos[1]
                    new_global_pos[1] += sin_yaw * rel_pos[0] + cos_yaw * rel_pos[1]
                    new_global_yaw = normalize_angle(self.robot_yaw + rel_yaw)
                
                self.map_manager.add_to_global_map(local_occupancy, new_global_pos, new_global_yaw)
            
            # 버퍼가 꽉 찬 경우: 첫 프레임 제거, robot_pos를 이전 두 번째 프레임으로 이동
            if len(self.point_cloud_buffer) == self.buffer_size:
                # frame_positions[1]은 버퍼 내 상대 좌표
                # 이를 현재 robot_pos 기준으로 전역 좌표로 변환
                rel_pos = frame_positions[1]
                rel_yaw = frame_yaws[1]
                
                cos_yaw = np.cos(self.robot_yaw)
                sin_yaw = np.sin(self.robot_yaw)
                
                # robot_pos를 버퍼의 두 번째 프레임 위치로 이동 (전역 좌표)
                new_robot_pos = self.robot_pos.copy()
                new_robot_pos[0] += cos_yaw * rel_pos[0] - sin_yaw * rel_pos[1]
                new_robot_pos[1] += sin_yaw * rel_pos[0] + cos_yaw * rel_pos[1]
                new_robot_yaw = normalize_angle(self.robot_yaw + rel_yaw)
                
                self.robot_pos = new_robot_pos
                self.robot_yaw = new_robot_yaw
                
                # 궤적에 추가
                self.robot_trajectory.append(self.robot_pos.copy())
                
                print(f"  Robot pos: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f}), "
                      f"yaw: {np.degrees(self.robot_yaw):.1f}°")
                
                print(f"  Robot pos: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f}), "
                      f"yaw: {np.degrees(self.robot_yaw):.1f}°")
        
        return True
    
    def get_current_robot_position(self):
        if len(self.buffer_positions) == 0:
            return self.robot_pos.copy()
        
        last_rel_pos = self.buffer_positions[-1]
        last_rel_yaw = self.buffer_yaws[-1]
        
        cos_yaw = np.cos(self.robot_yaw)
        sin_yaw = np.sin(self.robot_yaw)
        
        current_pos = self.robot_pos.copy()
        current_pos[0] += cos_yaw * last_rel_pos[0] - sin_yaw * last_rel_pos[1]
        current_pos[1] += sin_yaw * last_rel_pos[0] + cos_yaw * last_rel_pos[1]
        
        return current_pos
    
    def run(self, max_frames=None):
        all_frame_ids = self.get_file_list()
        
        if max_frames:
            all_frame_ids = all_frame_ids[1000:max_frames]
        
        all_frame_ids = all_frame_ids[::self.frame_interval]
        
        print(f"\nProcessing {len(all_frame_ids)} frames with frame_interval={self.frame_interval}, buffer_size={self.buffer_size}\n")
        
        for idx, frame_id in enumerate(all_frame_ids):
            success = self.process_frame(frame_id)
            
            if not success:
                continue
            
            rgb = self.load_frame(frame_id)
            if rgb is not None:
                current_pos = self.get_current_robot_position()
                self.visualizer.visualize(
                    frame_id, 
                    rgb, 
                    self.map_manager.global_map, 
                    current_pos, 
                    self.robot_trajectory, 
                    self.lidar_resolution
                )
            
            if (idx + 1) % 100 == 0:
                print(f"Progress: {idx + 1}/{len(all_frame_ids)}")
        
        print(f"\nDone! Visualizations saved to {self.vis_dir}")


if __name__ == "__main__":
    dataset_root = os.path.expanduser("~/agrivla/dataset/2025summer/site1-2/20250702_2105")
    project_root = os.path.expanduser("~/agrivla/navi")
    
    mapper = FarmMapper(dataset_root, project_root, buffer_size=4, frame_interval=30, num_random_pairs=2)
    mapper.run(max_frames=5000)