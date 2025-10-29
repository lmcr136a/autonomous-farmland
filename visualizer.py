import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


class MapVisualizer:
    def __init__(self, vis_dir, pixels_per_meter=20):
        self.vis_dir = Path(vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.pixels_per_meter = pixels_per_meter
    
    def render_occupancy_map(self, global_map, robot_pos):
        if len(global_map) == 0:
            img = np.ones((400, 400, 3), dtype=np.uint8) * 30
            return img, 200, 200, 0, 0, 400, 400
        
        world_xs = [k[0] for k in global_map.keys()]
        world_ys = [k[1] for k in global_map.keys()]
        
        min_x, max_x = min(world_xs), max(world_xs)
        min_y, max_y = min(world_ys), max(world_ys)
        
        min_x = min(min_x, robot_pos[0])
        max_x = max(max_x, robot_pos[0])
        min_y = min(min_y, robot_pos[1])
        max_y = max(max_y, robot_pos[1])
        
        margin = 5.0
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin
        
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        map_width = int(range_x * self.pixels_per_meter) + 1
        map_height = int(range_y * self.pixels_per_meter) + 1
        
        occupancy = np.ones((map_height, map_width, 3), dtype=np.uint8) * 30
        
        for (wx, wy), value in global_map.items():
            px = int((wx - min_x) * self.pixels_per_meter)
            py = int((wy - min_y) * self.pixels_per_meter)
            if 0 <= px < map_width and 0 <= py < map_height:
                if value == 0:
                    occupancy[py, px] = [255, 255, 255]
                else:
                    occupancy[py, px] = [0, 0, 255]
        
        robot_px = int((robot_pos[0] - min_x) * self.pixels_per_meter)
        robot_py = int((robot_pos[1] - min_y) * self.pixels_per_meter)
        
        return occupancy, robot_px, robot_py, min_x, min_y, map_width, map_height
    
    def visualize(self, frame_id, rgb, global_map, robot_pos, robot_trajectory, lidar_resolution):
        occupancy, robot_x, robot_y, min_x, min_y, map_w, map_h = self.render_occupancy_map(global_map, robot_pos)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        axes[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Frame: R_{frame_id}.png", fontsize=12)
        axes[0].axis('off')
        
        map_display = occupancy.copy()
        
        if 0 <= robot_x < map_w and 0 <= robot_y < map_h:
            cv2.circle(map_display, (robot_x, robot_y), 8, (0, 255, 0), -1)
            cv2.circle(map_display, (robot_x, robot_y), 10, (255, 255, 255), 2)
        
        axes[1].imshow(cv2.cvtColor(map_display, cv2.COLOR_BGR2RGB))
        
        if len(robot_trajectory) > 1:
            traj = np.array(robot_trajectory)
            traj_px = (traj[:, 0] - min_x) * self.pixels_per_meter
            traj_py = (traj[:, 1] - min_y) * self.pixels_per_meter
            axes[1].plot(traj_px, traj_py, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        
        if 0 <= robot_x < map_w and 0 <= robot_y < map_h:
            axes[1].plot(robot_x, robot_y, 'g*', markersize=20, label='Robot', 
                        markeredgecolor='white', markeredgewidth=1.5)
        
        axes[1].invert_yaxis()
        axes[1].set_title(f'Occupancy Map (White: Ground, Red: Obstacle, Dark: Unknown)', fontsize=12)
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        world_width = map_w / self.pixels_per_meter
        world_height = map_h / self.pixels_per_meter
        axes[1].text(0.02, 0.98, f'Map size: {world_width:.1f}m × {world_height:.1f}m\n'
                                  f'Robot: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})m',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        output_path = self.vis_dir / f"frame{int(frame_id):04d}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
        return map_display