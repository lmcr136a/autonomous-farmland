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


# def random_sample_points(points, num_samples=30000):
#     if len(points) <= num_samples:
#         return points
    
#     indices = np.random.choice(len(points), num_samples, replace=False)
#     return points[indices]


def random_sample_points(points, voxel_size=0.2, max_samples=30000):
    # points = np.array(points)
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    sampled_points = points[unique_indices]
    
    if len(sampled_points) > max_samples:
        indices = np.random.choice(len(sampled_points), max_samples, replace=False)
        sampled_points = sampled_points[indices]

    return sampled_points



class FarmMapper:
    def __init__(self, data_root, lidar_resolution=5, buffer_size=10, 
                 num_random_pairs=None, fitness_threshold=0.95, sample_points_num=30000,
                 cell_size=0.1, use_obstacles_only=False):
        self.data_root = Path(data_root)
        self.lidar_resolution = lidar_resolution
        self.buffer_size = buffer_size
        self.num_random_pairs = num_random_pairs if num_random_pairs is not None else buffer_size // 2
        self.fitness_threshold = fitness_threshold
        self.sample_points_num = sample_points_num
        self.cell_size = cell_size  # Cell size in meters (e.g., 0.1 = 10cm)
        self.use_obstacles_only = use_obstacles_only  # If True, use only obstacle points for ICP
        
        self.lidar_dir = self.data_root / "LiDAR"
        self.rgb_dir = self.data_root / "RGB-D"
        self.vis_dir = Path("./visualization")
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = MapVisualizer(self.vis_dir)
        
        self.robot_trajectory = [np.array([0.0, 0.0], dtype=np.float32)]
        self.robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_yaw = 0.0
        
        self.point_cloud_buffer = deque(maxlen=buffer_size)
        self.local_cell_map_buffer = deque(maxlen=buffer_size)
        self.buffer_positions = deque(maxlen=buffer_size)
        self.buffer_yaws = deque(maxlen=buffer_size)
        
        self.camera_intrinsics = self.estimate_camera_intrinsics()
        self.pose_calculator = PoseCalculator()
        self.map_manager = GlobalMapManager()
        
        self.frame_count = 0
        
        print(f"FarmMapper initialized")
        print(f"LiDAR resolution: {self.lidar_resolution}cm/cell")
        print(f"Cell size: {self.cell_size*100:.0f}cm ({self.cell_size}m)")
        print(f"Use obstacles only for ICP: {self.use_obstacles_only}")
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
    
    def load_rgb_from_png(self, png_path):
        for _ in range(50):
                        
            if not os.path.exists(png_path):
                time.sleep(1)
                continue
                        
            rgb = cv2.imread(str(png_path))
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            return rgb
        
        if not os.path.exists(png_path):
            print(f"{png_path} not exists.")
            exit()
    
        

    def load_lidar_from_bin(self, bin_path, min_distance=0.1, max_distance=50.0):
        for _ in range(50):
                
            if not os.path.exists(bin_path):
                time.sleep(1)
                continue
            
            points = np.fromfile(bin_path, dtype=np.float32)

            points = points[3:].reshape(-1, 4)/ 100.0

            transformed = np.zeros_like(points)
            transformed[:, 0] = points[:, 1]
            transformed[:, 1] = points[:, 0]
            transformed[:, 2] = -points[:, 2]
            transformed[:, 3] = points[:, 3]
            transformed = transformed[transformed[:,2] < 0]
            
            distances = np.sqrt(transformed[:, 0]**2 + transformed[:, 1]**2)
            distance_mask = distances <= max_distance
            robot_body_mask = ~((transformed[:, 1] >= -0.8) & (transformed[:, 1] <= 0.2) & 
                    (transformed[:, 0] >= -0.6) & (transformed[:, 0] <= 0.65))
            
            transformed = transformed[distance_mask & robot_body_mask]
            

            sampled_points = random_sample_points(transformed, max_samples=self.sample_points_num)
            sampled_points = np.array(sampled_points[:, :-1], dtype=np.float64)
            
            return sampled_points
        
        if not os.path.exists(bin_path):
            print(f"{bin_path} not exists.")
            exit()
            
            
    def points_to_cell_map(self, points, height_threshold=-0.5):
        """
        Convert point cloud to cell-based map
        Returns: dict with cell coordinates as keys and values: 0=unknown, 1=road, 2=obstacle
        """
        if len(points) == 0:
            return {}
        
        cell_map = {}
        
        # Group points by cell
        for p in points:
            # Calculate cell coordinates based on cell_size
            cell_x = int(np.floor(p[0] / self.cell_size))
            cell_y = int(np.floor(p[1] / self.cell_size))
            cell_key = (cell_x, cell_y)
            
            # Determine if this point is road or obstacle based on height
            if p[2] < height_threshold:
                point_type = 1  # road
            else:
                point_type = 2  # obstacle
            
            # Update cell: obstacles take priority over road
            if cell_key not in cell_map:
                cell_map[cell_key] = point_type
            elif cell_map[cell_key] == 1 and point_type == 2:
                # If cell was marked as road but we found an obstacle, update it
                cell_map[cell_key] = 2
        
        return cell_map
    
    def filter_obstacle_points(self, points, height_threshold=-0.5):
        """
        Filter only obstacle points (above height_threshold) for ICP
        Returns: obstacle points only
        """
        if len(points) == 0:
            return points
        
        # Keep only points above the height threshold (obstacles)
        obstacle_mask = points[:, 2] >= height_threshold
        obstacle_points = points[obstacle_mask]
        
        print(f"    Filtered: {len(points)} total -> {len(obstacle_points)} obstacles ({len(obstacle_points)/len(points)*100:.1f}%)")
        
        return obstacle_points
    
    def process_files(self, bin_path, rgb_path):

        lidar = self.load_lidar_from_bin(bin_path)
        rgb = self.load_rgb_from_png(rgb_path)
        
        
        local_cell_map = self.points_to_cell_map(lidar)
        
        # For ICP: use obstacle points only if enabled, otherwise use all points
        if self.use_obstacles_only:
            lidar_for_icp = self.filter_obstacle_points(lidar)
        else:
            lidar_for_icp = lidar.copy()
        
        temp_buffer = list(self.point_cloud_buffer) + [lidar_for_icp]
        
        if len(self.buffer_positions) == 0:
            if self.use_obstacles_only:
                lidar_for_icp = self.filter_obstacle_points(lidar)
            else:
                lidar_for_icp = lidar.copy()
                
            self.point_cloud_buffer.append(lidar_for_icp)
            self.local_cell_map_buffer.append(local_cell_map)
            self.buffer_positions.append(np.array([0.0, 0.0]))
            self.buffer_yaws.append(0.0)
            self.frame_count += 1
            os.remove(bin_path)
            os.remove(rgb_path)
            print(f"  First frame accepted")
            return True
        
        random_pairs = generate_random_pairs(len(temp_buffer)-1, self.num_random_pairs)
        
        result = self.pose_calculator.estimate_motion_multiframe(
            temp_buffer, 
            pairs=random_pairs,
            current_global_pos=self.robot_pos,
            current_global_yaw=self.robot_yaw
        )
        
        os.remove(bin_path)
        os.remove(rgb_path)
        print("Data loaded and deleted")
        
        if result is not None:
            frame_positions, frame_yaws, global_positions, global_yaws = result
            
            if self.use_obstacles_only:
                lidar_for_icp = self.filter_obstacle_points(lidar)
            else:
                lidar_for_icp = lidar.copy()
                
            self.point_cloud_buffer.append(lidar_for_icp)
            self.local_cell_map_buffer.append(local_cell_map)
            buffer_positions_old = [pos.copy() for pos in self.buffer_positions]
            buffer_yaws_old = list(self.buffer_yaws)
                    
            positions_changed = []
            for i in range(len(self.buffer_positions)):
                old_pos = self.buffer_positions[i]
                new_pos = frame_positions[i]
                old_yaw = self.buffer_yaws[i]
                new_yaw = frame_yaws[i]
                
                pos_diff = np.linalg.norm(old_pos - new_pos)
                yaw_diff = abs(normalize_angle(new_yaw - old_yaw))
                
                if pos_diff > 0.05 or yaw_diff > np.deg2rad(0.4):
                    positions_changed.append(i)
                    print(f"    Frame {i} updated: "
                        f"pos Δ{pos_diff:.3f}m, yaw Δ{np.degrees(yaw_diff):.1f}°")

            for i in range(len(frame_positions)):
                if i < len(self.buffer_positions):
                    self.buffer_positions[i] = frame_positions[i].copy()
                    self.buffer_yaws[i] = frame_yaws[i]
                else:
                    self.buffer_positions.append(frame_positions[i].copy())
                    self.buffer_yaws.append(frame_yaws[i])
                    
            if len(positions_changed) > 0:
                
                print(f"  {len(positions_changed)} frame(s) updated, rebuilding global map...")
                self.map_manager.update_global_map_for_buffer(
                    frame_positions, frame_yaws,
                    buffer_positions_old, buffer_yaws_old,  
                    self.local_cell_map_buffer,
                    self.robot_pos, self.robot_yaw,
                    self.cell_size
                )
                
                if len(self.robot_trajectory) >= len(self.buffer_positions):
                    trajectory_start_idx = len(self.robot_trajectory) - len(self.buffer_positions)
                                
                    for i in range(len(self.buffer_positions)):
                        rel_pos = self.buffer_positions[i]
                        rel_yaw = self.buffer_yaws[i]
                        cos_yaw = np.cos(self.robot_yaw)
                        sin_yaw = np.sin(self.robot_yaw)
                        global_pos = self.robot_pos.copy()
                        global_pos[0] += cos_yaw * rel_pos[0] - sin_yaw * rel_pos[1]
                        global_pos[1] += sin_yaw * rel_pos[0] + cos_yaw * rel_pos[1]
                        
                        self.robot_trajectory[trajectory_start_idx + i] = global_pos
            
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
                
                self.map_manager.add_to_global_map(local_cell_map, new_global_pos, new_global_yaw, self.cell_size)
            
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
                
                print(f"{str('~')*100}Robot pos: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f}), "
                        f"yaw: {np.degrees(self.robot_yaw):.1f}°")
            
            if rgb is not None:
                current_pos = self.get_current_robot_position()
                self.visualizer.visualize(
                    self.frame_count, 
                    rgb, 
                    self.map_manager.global_map, 
                    current_pos, 
                    self.robot_trajectory, 
                    self.cell_size
                )
            
            self.frame_count += 1
            
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
    
    def run_realtime(self, check_interval=1.0):
        print(f"\nStarting realtime processing...")
        print(f"Checking for new files every {check_interval} seconds")
        bin_path, rgb_path = os.path.join(self.lidar_dir, "pointcloud_sync.bin"), os.path.join(self.rgb_dir, "rgb.png")
        
        print(f"\nProcessing files:")
        print(f"  BIN: {bin_path}")
        print(f"  RGB: {rgb_path}")
        
        
        try:
            while True:
                
                success = self.process_files(bin_path, rgb_path)
                
                if success:
                    print(f"  ✓ Files processed successfully!")
                else:
                    print(f"  ✗ Failed to process files")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n\nStopped by user")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Visualizations saved to {self.vis_dir}")


if __name__ == "__main__":
    data_root = "/home/nahyeon/navi/AgriChrono/data/fargo/1110"
    subdirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    data_root = os.path.join(data_root, subdirs[-1])
    
    mapper = FarmMapper(
        data_root,
        buffer_size=10, 
        num_random_pairs=4, 
        fitness_threshold=0.85,
        sample_points_num=10000,
        cell_size=0.05,  # 10cm cells
        use_obstacles_only=True  # Set to True to use only obstacle points for ICP
    )
    
    mapper.run_realtime(check_interval=1.0)