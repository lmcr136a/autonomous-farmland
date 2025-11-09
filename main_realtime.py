import os
import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
from visualizer import MapVisualizer
from collections import deque
from cal_pose import PoseCalculator, GlobalMapManager, normalize_angle
import random
import time


def generate_random_pairs(buffer_size, num_pairs=None):
    if num_pairs is None:
        num_pairs = buffer_size // 2
    
    pairs = []
    
    for i in range(buffer_size):
        possible_prev = []
        if i >= 1:
            possible_prev.append(i - 1)
        if i >= 2:
            possible_prev.append(i - 2)
        
        if len(possible_prev) > 0:
            prev_idx = random.choice(possible_prev)
            pairs.append((prev_idx, i))
    
    pairs = list(set(pairs))
    random.shuffle(pairs)
    
    return pairs[:num_pairs]


def random_sample_points(points, num_samples=30000):
    if len(points) <= num_samples:
        return points
    
    indices = np.random.choice(len(points), num_samples, replace=False)
    return points[indices]


class FarmMapper:
    def __init__(self, data_root, lidar_resolution=5, buffer_size=10, 
                 num_random_pairs=None, fitness_threshold=0.95, sample_points_num=30000):
        self.data_root = Path(data_root)
        self.lidar_resolution = lidar_resolution
        self.buffer_size = buffer_size
        self.num_random_pairs = num_random_pairs if num_random_pairs is not None else buffer_size // 2
        self.fitness_threshold = fitness_threshold
        self.sample_points_num = sample_points_num
        
        self.lidar_dir = self.data_root / "LiDAR"
        self.rgb_dir = self.data_root / "RGB-D"
        self.vis_dir = self.data_root / "visualization"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = MapVisualizer(self.vis_dir)
        
        self.robot_trajectory = [np.array([0.0, 0.0], dtype=np.float32)]
        self.robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_yaw = 0.0
        
        self.point_cloud_buffer = deque(maxlen=buffer_size)
        self.local_occupancy_buffer = deque(maxlen=buffer_size)
        self.buffer_positions = deque(maxlen=buffer_size)
        self.buffer_yaws = deque(maxlen=buffer_size)
        
        self.camera_intrinsics = self.estimate_camera_intrinsics()
        self.pose_calculator = PoseCalculator()
        self.map_manager = GlobalMapManager()
        
        self.frame_count = 0
        
        print(f"FarmMapper initialized")
        print(f"LiDAR resolution: {self.lidar_resolution}cm/cell")
        print(f"Window size: {self.buffer_size} frames")
        print(f"Random pairs per window: {self.num_random_pairs}")
        print(f"Fitness threshold: {self.fitness_threshold}")
        print(f"Sample points per frame: {self.sample_points_num}")
    
    def estimate_camera_intrinsics(self):
        fx = 1258.97
        fy = 1258.97
        cx = 916.48
        cy = 553.83
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    def find_latest_files(self):
        bin_files = sorted(self.lidar_dir.glob("*.bin"))
        rgb_files = sorted(self.rgb_dir.glob("rgb.png"))
        
        if len(bin_files) > 0 and len(rgb_files) > 0:
            return bin_files[-1], rgb_files[-1]
        return None, None
    
    def load_rgb_from_png(self, png_path):
        if not png_path.exists():
            return None
        
        rgb = cv2.imread(str(png_path))
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        return rgb
    
    def load_lidar_from_bin(self, bin_path, min_distance=2, max_distance=30.0):
        if not bin_path.exists():
            return None
        
        with open(bin_path, 'rb') as f:
            data = f.read()
        
        file_size = len(data)
        
        if file_size % 16 == 0:
            points = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
            xyz = points[:, :3]
        elif file_size % 12 == 0:
            points = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
            xyz = points[:, :3]
        elif file_size % 20 == 0:
            points = np.frombuffer(data, dtype=np.float32).reshape(-1, 5)
            xyz = points[:, :3]
        else:
            points = np.frombuffer(data, dtype=np.float32)
            num_points = len(points) // 3
            xyz = points[:num_points*3].reshape(-1, 3)
        
        points_m = xyz / 100.0
        
        transformed = np.zeros_like(points_m)
        transformed[:, 0] = -points_m[:, 1]
        transformed[:, 1] = points_m[:, 0]
        transformed[:, 2] = points_m[:, 2]
        
        distances = np.sqrt(transformed[:, 0]**2 + transformed[:, 1]**2)
        valid_points = transformed[(distances >= min_distance) & (distances <= max_distance)]
        
        sampled_points = random_sample_points(valid_points, self.sample_points_num)
        
        return sampled_points
    
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
    
    def process_files(self, bin_path, rgb_path):
        print(f"\nProcessing files:")
        print(f"  BIN: {bin_path.name}")
        print(f"  RGB: {rgb_path.name}")
        
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            lidar = self.load_lidar_from_bin(bin_path)
            rgb = self.load_rgb_from_png(rgb_path)
            
            if lidar is None:
                print("  No LiDAR data available")
                return False
            
            if len(lidar) < 100:
                print("  Too few points")
                return False
            
            local_occupancy = self.points_to_ground_occupancy(lidar)
            
            temp_buffer = list(self.point_cloud_buffer) + [lidar.copy()]
            
            if len(self.buffer_positions) == 0:
                self.point_cloud_buffer.append(lidar.copy())
                self.local_occupancy_buffer.append(local_occupancy)
                self.buffer_positions.append(np.array([0.0, 0.0]))
                self.buffer_yaws.append(0.0)
                print(f"  First frame accepted")
                
                if rgb is not None:
                    current_pos = self.get_current_robot_position()
                    self.visualizer.visualize(
                        self.frame_count, 
                        rgb, 
                        self.map_manager.global_map, 
                        current_pos, 
                        self.robot_trajectory, 
                        self.lidar_resolution
                    )
                
                self.frame_count += 1
                bin_path.unlink()
                rgb_path.unlink()
                return True
            
            random_pairs = generate_random_pairs(len(temp_buffer), self.num_random_pairs)
            print(f"  Attempt {attempt + 1}: Random pairs: {random_pairs}")
            
            result = self.pose_calculator.estimate_motion_multiframe(
                temp_buffer, 
                refine_rotation=True,
                random_pairs=random_pairs,
                current_global_pos=self.robot_pos,
                current_global_yaw=self.robot_yaw
            )
            
            if result is None:
                attempt += 1
                print(f"  Registration failed, retrying... (attempt {attempt}/{max_attempts})")
                continue
            
            if len(result) == 5:
                frame_positions, frame_yaws, global_positions, global_yaws, fitness = result
            elif len(result) == 4:
                frame_positions, frame_yaws, global_positions, global_yaws = result
                fitness = None
            else:
                frame_positions, frame_yaws = result
                global_positions, global_yaws = None, None
                fitness = None
            
            if fitness is not None:
                print(f"  Fitness: {fitness:.4f} (threshold: {self.fitness_threshold})")
                
                if fitness < self.fitness_threshold:
                    attempt += 1
                    print(f"  Fitness too low, retrying... (attempt {attempt}/{max_attempts})")
                    continue
                else:
                    print(f"  ✓ Fitness acceptable!")
            
            self.point_cloud_buffer.append(lidar.copy())
            self.local_occupancy_buffer.append(local_occupancy)
            
            positions_changed = []
            
            for i in range(len(self.buffer_positions)):
                old_pos = self.buffer_positions[i]
                new_pos = frame_positions[i]
                old_yaw = self.buffer_yaws[i]
                new_yaw = frame_yaws[i]
                
                pos_diff = np.linalg.norm(old_pos - new_pos)
                yaw_diff = abs(normalize_angle(new_yaw - old_yaw))
                
                if pos_diff > 0.05 or yaw_diff > np.deg2rad(2):
                    positions_changed.append(i)
                    print(f"    Frame {i} updated: "
                          f"pos ({old_pos[0]:.2f}, {old_pos[1]:.2f}) -> ({new_pos[0]:.2f}, {new_pos[1]:.2f}), "
                          f"yaw {np.degrees(old_yaw):.1f}° -> {np.degrees(new_yaw):.1f}°")
            
            if len(positions_changed) > 0:
                print(f"  {len(positions_changed)} frame(s) updated, rebuilding global map...")
                
                self.map_manager.update_global_map_for_buffer(
                    frame_positions, frame_yaws, 
                    self.buffer_positions, self.buffer_yaws,
                    self.local_occupancy_buffer, self.robot_pos, self.robot_yaw
                )
                
                for i in range(len(frame_positions)):
                    if i < len(self.buffer_positions):
                        self.buffer_positions[i] = frame_positions[i].copy()
                        self.buffer_yaws[i] = frame_yaws[i]
                    else:
                        self.buffer_positions.append(frame_positions[i].copy())
                        self.buffer_yaws.append(frame_yaws[i])
                
                if global_positions is not None and len(self.robot_trajectory) > 0:
                    self.robot_trajectory[-1] = global_positions[0].copy()
                    print(f"  Updated trajectory point: ({global_positions[0][0]:.2f}, {global_positions[0][1]:.2f})")
            else:
                self.buffer_positions.append(frame_positions[-1].copy())
                self.buffer_yaws.append(frame_yaws[-1])
                
                if global_positions is not None and global_yaws is not None:
                    new_global_pos = global_positions[-1]
                    new_global_yaw = global_yaws[-1]
                else:
                    rel_pos = frame_positions[-1]
                    rel_yaw = frame_yaws[-1]
                    
                    cos_yaw = np.cos(self.robot_yaw)
                    sin_yaw = np.sin(self.robot_yaw)
                    
                    new_global_pos = self.robot_pos.copy()
                    new_global_pos[0] += cos_yaw * rel_pos[0] - sin_yaw * rel_pos[1]
                    new_global_pos[1] += sin_yaw * rel_pos[0] + cos_yaw * rel_pos[1]
                    new_global_yaw = normalize_angle(self.robot_yaw + rel_yaw)
                
                self.map_manager.add_to_global_map(local_occupancy, new_global_pos, new_global_yaw)
            
            if len(self.point_cloud_buffer) == self.buffer_size:
                rel_pos = frame_positions[1]
                rel_yaw = frame_yaws[1]
                
                cos_yaw = np.cos(self.robot_yaw)
                sin_yaw = np.sin(self.robot_yaw)
                
                new_robot_pos = self.robot_pos.copy()
                new_robot_pos[0] += cos_yaw * rel_pos[0] - sin_yaw * rel_pos[1]
                new_robot_pos[1] += sin_yaw * rel_pos[0] + cos_yaw * rel_pos[1]
                new_robot_yaw = normalize_angle(self.robot_yaw + rel_yaw)
                
                self.robot_pos = new_robot_pos
                self.robot_yaw = new_robot_yaw
                
                self.robot_trajectory.append(self.robot_pos.copy())
                
                print(f"  Robot pos: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f}), "
                      f"yaw: {np.degrees(self.robot_yaw):.1f}°")
            
            if rgb is not None:
                current_pos = self.get_current_robot_position()
                self.visualizer.visualize(
                    self.frame_count, 
                    rgb, 
                    self.map_manager.global_map, 
                    current_pos, 
                    self.robot_trajectory, 
                    self.lidar_resolution
                )
            
            self.frame_count += 1
            
            print(f"  Deleting processed files...")
            bin_path.unlink()
            rgb_path.unlink()
            
            return True
        
        print(f"  ✗ Failed to achieve fitness threshold after {max_attempts} attempts")
        return False
    
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
    
    def run_realtime(self, check_interval=1.0):
        print(f"\nStarting realtime processing...")
        print(f"Checking for new files every {check_interval} seconds")
        print(f"Press Ctrl+C to stop\n")
        
        processed_bins = set()
        processed_rgbs = set()
        
        try:
            while True:
                bin_path, rgb_path = self.find_latest_files()
                
                if bin_path is not None and rgb_path is not None:
                    if bin_path not in processed_bins and rgb_path not in processed_rgbs:
                        success = self.process_files(bin_path, rgb_path)
                        
                        if success:
                            processed_bins.add(bin_path)
                            processed_rgbs.add(rgb_path)
                            print(f"  ✓ Files processed successfully!")
                        else:
                            print(f"  ✗ Failed to process files")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n\nStopped by user")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Visualizations saved to {self.vis_dir}")


if __name__ == "__main__":
    data_root = "/home/nahyeon/navi/AgriChrono/data/fargo/test300/20251105_1623"
    
    mapper = FarmMapper(
        data_root,
        buffer_size=4, 
        num_random_pairs=2, 
        fitness_threshold=0.95,
        sample_points_num=30000
    )
    
    mapper.run_realtime(check_interval=1.0)