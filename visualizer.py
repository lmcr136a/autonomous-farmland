import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


class MapVisualizer:
    def __init__(self, vis_dir, pixels_per_meter=20):
        self.vis_dir = Path(vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.pixels_per_meter = pixels_per_meter
    
    def render_cell_map(self, cell_map, robot_pos, cell_size):
        """
        Render cell-based occupancy map
        cell_map: dict with (cell_x, cell_y) as keys and 0/1/2 as values
        cell_size: size of each cell in meters
        Returns: rendered image and coordinate info
        """
        if len(cell_map) == 0:
            img = np.ones((400, 400, 3), dtype=np.uint8) * 30
            return img, 200, 200, 0, 0, 400, 400
        
        # Get cell coordinate bounds
        cell_xs = [k[0] for k in cell_map.keys()]
        cell_ys = [k[1] for k in cell_map.keys()]
        
        # Calculate robot's cell position
        robot_cell_x = int(np.floor(robot_pos[0] / cell_size))
        robot_cell_y = int(np.floor(robot_pos[1] / cell_size))
        
        min_cell_x = min(min(cell_xs), robot_cell_x)
        max_cell_x = max(max(cell_xs), robot_cell_x)
        min_cell_y = min(min(cell_ys), robot_cell_y)
        max_cell_y = max(max(cell_ys), robot_cell_y)
        
        # Add margin in cells
        margin_cells = int(np.ceil(5.0 / cell_size))  # 5m margin
        min_cell_x -= margin_cells
        min_cell_y -= margin_cells
        max_cell_x += margin_cells
        max_cell_y += margin_cells
        
        # Calculate map dimensions in cells
        map_width_cells = max_cell_x - min_cell_x + 1
        map_height_cells = max_cell_y - min_cell_y + 1
        
        # Each cell will be rendered as multiple pixels for visibility
        pixels_per_cell = max(1, int(self.pixels_per_meter * cell_size))
        
        map_width_pixels = map_width_cells * pixels_per_cell
        map_height_pixels = map_height_cells * pixels_per_cell
        
        # Initialize map with dark background (unknown)
        cell_img = np.ones((map_height_pixels, map_width_pixels, 3), dtype=np.uint8) * 30
        
        # Render each cell
        for (cell_x, cell_y), value in cell_map.items():
            # Calculate pixel position for this cell
            px_start = (cell_x - min_cell_x) * pixels_per_cell
            py_start = (cell_y - min_cell_y) * pixels_per_cell
            px_end = px_start + pixels_per_cell
            py_end = py_start + pixels_per_cell
            
            if 0 <= px_start < map_width_pixels and 0 <= py_start < map_height_pixels:
                if value == 1:  # road
                    cell_img[py_start:py_end, px_start:px_end] = [255, 255, 255]
                elif value == 2:  # obstacle
                    cell_img[py_start:py_end, px_start:px_end] = [0, 0, 255]
                # value == 0 (unknown) stays as dark background
        
        # Calculate robot pixel position
        robot_px = int((robot_cell_x - min_cell_x + 0.5) * pixels_per_cell)
        robot_py = int((robot_cell_y - min_cell_y + 0.5) * pixels_per_cell)
        
        # Convert cell bounds back to world coordinates for display
        min_x = min_cell_x * cell_size
        min_y = min_cell_y * cell_size
        
        return cell_img, robot_px, robot_py, min_x, min_y, map_width_pixels, map_height_pixels
    
    def visualize(self, frame_id, rgb, cell_map, robot_pos, robot_trajectory, cell_size):
        """
        Visualize RGB image and cell-based map
        cell_size: size of each cell in meters
        """
        cell_img, robot_x, robot_y, min_x, min_y, map_w, map_h = self.render_cell_map(
            cell_map, robot_pos, cell_size
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        axes[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Frame: R_{frame_id}.png", fontsize=12)
        axes[0].axis('off')
        
        map_display = cell_img.copy()
        
        if 0 <= robot_x < map_w and 0 <= robot_y < map_h:
            cv2.circle(map_display, (robot_x, robot_y), 8, (0, 255, 0), -1)
            cv2.circle(map_display, (robot_x, robot_y), 10, (255, 255, 255), 2)
        
        axes[1].imshow(cv2.cvtColor(map_display, cv2.COLOR_BGR2RGB))
        
        if len(robot_trajectory) > 1:
            traj = np.array(robot_trajectory)
            # Convert world coordinates to pixel coordinates
            # First convert to cell coordinates, then to pixels
            traj_cell_x = np.floor(traj[:, 0] / cell_size).astype(int)
            traj_cell_y = np.floor(traj[:, 1] / cell_size).astype(int)
            
            # Convert cell coordinates to pixel coordinates
            min_cell_x = int(np.floor(min_x / cell_size))
            min_cell_y = int(np.floor(min_y / cell_size))
            pixels_per_cell = max(1, int(self.pixels_per_meter * cell_size))
            
            traj_px = (traj_cell_x - min_cell_x + 0.5) * pixels_per_cell
            traj_py = (traj_cell_y - min_cell_y + 0.5) * pixels_per_cell
            
            axes[1].plot(traj_px, traj_py, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        
        if 0 <= robot_x < map_w and 0 <= robot_y < map_h:
            axes[1].plot(robot_x, robot_y, 'g*', markersize=20, label='Robot', 
                        markeredgecolor='white', markeredgewidth=1.5)
        
        axes[1].invert_yaxis()
        axes[1].set_title(f'Cell Map (White: Road, Red: Obstacle, Dark: Unknown)', fontsize=12)
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        world_width = map_w * cell_size / self.pixels_per_meter
        world_height = map_h * cell_size / self.pixels_per_meter
        num_cells = len(cell_map)
        axes[1].text(0.02, 0.98, 
                    f'Map size: {world_width:.1f}m × {world_height:.1f}m\n'
                    f'Cell size: {cell_size*100:.0f}cm\n'
                    f'Active cells: {num_cells}\n'
                    f'Robot: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})m',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        output_path = self.vis_dir / f"frame{int(frame_id):04d}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
        return map_display