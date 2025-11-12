import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

RADIUS = 0.5
GICP_NN = 50
VOXEL_SIZE = 0.05
FITNESS = 0.85

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class PlaneDetector:
    
    def __init__(self, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
    
    def detect_planes(self, points, min_plane_size=500, max_planes=10):
        if len(points) < 100:
            return []
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        planes = []
        remaining_points = points.copy()
        
        for _ in range(max_planes):
            if len(remaining_points) < min_plane_size:
                break
            
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(remaining_points)
            
            plane_model, inliers = pcd_temp.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )
            
            if len(inliers) < min_plane_size:
                break
            
            inlier_points = remaining_points[inliers]
            
            a, b, c, d = plane_model
            plane_normal = np.array([a, b, c])
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            
            plane_center = np.mean(inlier_points, axis=0)
            
            plane_type = self._classify_plane(plane_normal)
            
            plane_bounds = self._compute_plane_bounds(inlier_points, plane_normal)
            
            planes.append({
                'model': plane_model,
                'normal': plane_normal,
                'center': plane_center,
                'points': inlier_points,
                'type': plane_type,
                'bounds': plane_bounds,
                'size': len(inlier_points)
            })
            
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False
            remaining_points = remaining_points[mask]
        
        return planes
    
    def _classify_plane(self, normal):
        abs_normal = np.abs(normal)
        
        if abs_normal[2] > 0.9:
            return 'horizontal'
        elif abs_normal[0] > 0.7 or abs_normal[1] > 0.7:
            return 'vertical'
        else:
            return 'angled'
    
    def _compute_plane_bounds(self, points, normal):
        u = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        u = u - np.dot(u, normal) * normal
        u = u / np.linalg.norm(u)
        
        v = np.cross(normal, u)
        
        coords_u = np.dot(points, u)
        coords_v = np.dot(points, v)
        
        return {
            'u_min': coords_u.min(),
            'u_max': coords_u.max(),
            'v_min': coords_v.min(),
            'v_max': coords_v.max(),
            'area': (coords_u.max() - coords_u.min()) * (coords_v.max() - coords_v.min())
        }
    
    def match_planes(self, planes1, planes2, angle_threshold=10.0, distance_threshold=0.5):
        angle_threshold_rad = np.deg2rad(angle_threshold)
        matches = []
        
        for i, p1 in enumerate(planes1):
            best_match = None
            best_score = -1
            
            for j, p2 in enumerate(planes2):
                if p1['type'] != p2['type']:
                    continue
                
                normal_angle = np.arccos(np.clip(np.abs(np.dot(p1['normal'], p2['normal'])), 0, 1))
                
                if normal_angle > angle_threshold_rad:
                    continue
                
                center_distance = np.linalg.norm(p1['center'] - p2['center'])
                
                if center_distance > distance_threshold * 5:
                    continue
                
                size_ratio = min(p1['size'], p2['size']) / max(p1['size'], p2['size'])
                
                area_ratio = min(p1['bounds']['area'], p2['bounds']['area']) / max(p1['bounds']['area'], p2['bounds']['area'])
                
                score = size_ratio * 0.4 + area_ratio * 0.3 + (1 - normal_angle / angle_threshold_rad) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_match = j
            
            if best_match is not None and best_score > 0.5:
                matches.append((i, best_match, best_score))
        
        return matches


class PoseCalculator:
    def __init__(self):
        self.cached_normals = {}
        self.plane_detector = PlaneDetector(
            distance_threshold=0.05,
            ransac_n=3,
            num_iterations=1000
        )
    
    def downsample_points(self, points, voxel_size=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd_down.points, dtype=np.float64)
    
    def estimate_rotation_from_planes(self, planes1, planes2, matches):
        if len(matches) < 2:
            return None, 0.0
        
        vertical_planes1 = []
        vertical_planes2 = []
        
        for i, j, score in matches:
            if planes1[i]['type'] == 'vertical':
                vertical_planes1.append(planes1[i])
                vertical_planes2.append(planes2[j])
        
        if len(vertical_planes1) < 2:
            print("No vertical planes")
            return None, 0.0
        
        angles = []
        weights = []
        
        for p1, p2 in zip(vertical_planes1, vertical_planes2):
            n1_2d = p1['normal'][:2]
            n2_2d = p2['normal'][:2]
            
            n1_2d = n1_2d / np.linalg.norm(n1_2d)
            n2_2d = n2_2d / np.linalg.norm(n2_2d)
            
            cos_angle = np.dot(n1_2d, n2_2d)
            sin_angle = np.cross(n1_2d, n2_2d)
            
            angle = np.arctan2(sin_angle, cos_angle)
            
            angles.append(angle)
            weights.append(p1['size'])
        
        angles = np.array(angles)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        mean_angle = np.arctan2(
            np.sum(weights * np.sin(angles)),
            np.sum(weights * np.cos(angles))
        )
        
        angle_std = np.sqrt(np.sum(weights * (angles - mean_angle) ** 2))
        confidence = np.exp(-angle_std / np.deg2rad(10))
        
        confidence *= min(len(vertical_planes1) / 3.0, 1.0)
        
        return mean_angle, confidence
    
    def estimate_translation_from_planes(self, planes1, planes2, matches, rotation_angle):
        if len(matches) == 0:
            return None, 0.0
        
        translations = []
        weights = []
        
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        R_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        for i, j, score in matches:
            c1 = planes1[i]['center']
            c2 = planes2[j]['center']
            
            c2_rotated = c2.copy()
            c2_rotated[:2] = R_2d @ c2[:2]
            
            translation = c1 - c2_rotated
            
            translations.append(translation)
            weights.append(planes1[i]['size'] * score)
        
        translations = np.array(translations)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        mean_translation = np.sum(translations * weights[:, np.newaxis], axis=0)
        
        translation_std = np.sqrt(np.sum(weights[:, np.newaxis] * (translations - mean_translation) ** 2, axis=0))
        confidence = np.exp(-np.linalg.norm(translation_std) / 0.5)
        
        return mean_translation, confidence
    

    def estimate_rotation_robust(self, prev_points, curr_points):
        prev_down = self.downsample_points(prev_points, voxel_size=0.15)
        curr_down = self.downsample_points(curr_points, voxel_size=0.15)
        
        if len(prev_down) < 100 or len(curr_down) < 100:
            return 0.0, 0.0
        
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(curr_down)
        target.points = o3d.utility.Vector3dVector(prev_down)
        
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
        
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source, o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=100))
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=0.75,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.75)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        R = result.transformation[:3, :3]
        angle = np.arctan2(R[1,0], R[0,0])
        confidence = result.fitness
        
        print(f"    FPFH-RANSAC: angle={np.degrees(angle):.2f}°, fitness={confidence:.3f}")
        
        return angle, confidence
    
    def estimate_rotation_polar_fast(self, prev_points, curr_points, num_angles=360, 
                                     angle_range=None, center_angle=None):
        prev_2d = prev_points[:, :2]
        curr_2d = curr_points[:, :2]
        
        n_prev = len(prev_2d)
        n_curr = len(curr_2d)
        sample_size_prev = min(8000, n_prev)
        sample_size_curr = min(8000, n_curr)
        
        if n_prev > sample_size_prev:
            prev_indices = np.random.choice(n_prev, sample_size_prev, replace=False)
            prev_2d = prev_2d[prev_indices]
        
        if n_curr > sample_size_curr:
            curr_indices = np.random.choice(n_curr, sample_size_curr, replace=False)
            curr_2d = curr_2d[curr_indices]
        
        prev_center = prev_2d.mean(axis=0)
        curr_center = curr_2d.mean(axis=0)
        
        tree = KDTree(prev_2d)
        
        if angle_range is not None and center_angle is not None:
            angles_to_test = np.linspace(
                center_angle - angle_range/2, 
                center_angle + angle_range/2, 
                num_angles, 
                endpoint=True
            )
        else:
            angles_to_test = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)
        
        cos_angles = np.cos(angles_to_test)
        sin_angles = np.sin(angles_to_test)
        
        curr_centered = curr_2d - curr_center
        
        best_score = -np.inf
        best_angle = 0.0
        best_inlier_ratio = 0.0
        
        distances_in_prev = []
        for i in range(min(1000, len(prev_2d))):
            dist, _ = tree.query([prev_2d[i]], k=2)
            distances_in_prev.append(dist[0][1])
        median_spacing = np.median(distances_in_prev)
        inlier_threshold = max(0.25, median_spacing * 1.5)
        
        for i in range(num_angles):
            curr_rotated = np.empty_like(curr_centered)
            curr_rotated[:, 0] = cos_angles[i] * curr_centered[:, 0] - sin_angles[i] * curr_centered[:, 1]
            curr_rotated[:, 1] = sin_angles[i] * curr_centered[:, 0] + cos_angles[i] * curr_centered[:, 1]
            curr_rotated += prev_center
            
            distances, _ = tree.query(curr_rotated, k=1)
            
            inliers = distances < inlier_threshold
            inlier_ratio = np.sum(inliers) / len(distances)
            
            huber_delta = 1.0
            residuals = distances.copy()
            outliers = residuals > huber_delta
            residuals[outliers] = huber_delta * (2 * np.sqrt(residuals[outliers]) - 1)
            
            mean_inlier_dist = np.mean(distances[inliers]) if np.sum(inliers) > 0 else 1e6
            score = inlier_ratio * 100 - mean_inlier_dist * 10
            
            if score > best_score:
                best_score = score
                best_angle = angles_to_test[i]
                best_inlier_ratio = inlier_ratio
        
        confidence = best_inlier_ratio
        
        return best_angle, confidence
    
    def estimate_rotation_combined(self, prev_points, curr_points, refine=False, coarse_angle=None):
        if not refine or coarse_angle is None:
            # rotation_polar, conf_polar = self.estimate_rotation_polar_fast(
            #     prev_points, curr_points, num_angles=360, angle_range=np.deg2rad(180),
            # )
            
            rotation_polar, conf_polar = self.estimate_rotation_robust(
                                                        prev_points, curr_points)
            return rotation_polar, conf_polar
        else:
            angle_range = np.deg2rad(10)
            # rotation_fine, conf_fine = self.estimate_rotation_polar_fast(
            #     prev_points, curr_points, 
            #     num_angles=50,
            #     angle_range=angle_range,
            #     center_angle=coarse_angle
            # )
            
            rotation_fine, conf_fine = self.estimate_rotation_robust(
                                                        prev_points, curr_points)
            print(f"    Fine-tuning: {np.degrees(coarse_angle):.2f}° -> {np.degrees(rotation_fine):.2f}°")
            return rotation_fine, conf_fine
    

    def estimate_translation_icp(self, prev_points, curr_points, rotation_angle):
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        R_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        
        curr_2d = curr_points[:, :2]
        curr_center = curr_2d.mean(axis=0)
        curr_centered = curr_2d - curr_center
        curr_rotated_2d = (curr_centered @ R_2d.T) + curr_center
        
        curr_rotated_3d = curr_points.copy()
        curr_rotated_3d[:, :2] = curr_rotated_2d
        
        prev_down = self.downsample_points(prev_points, voxel_size=VOXEL_SIZE)
        curr_down = self.downsample_points(curr_rotated_3d, voxel_size=VOXEL_SIZE)
        
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(curr_down)
        target.points = o3d.utility.Vector3dVector(prev_down)
        
        # normals 필요 (GICP도 이걸 사용)
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS, max_nn=GICP_NN)
        )
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS, max_nn=GICP_NN)
        )
        
        source.orient_normals_consistent_tangent_plane(k=15)
        target.orient_normals_consistent_tangent_plane(k=15)
        
        # GICP로 변경: registration_generalized_icp + TransformationEstimationForGeneralizedICP
        max_corr_distance = 0.4  # 필요에 따라 0.2~0.6 사이로 튜닝
        trans_init = np.eye(4)
        
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-5,
            relative_rmse=1e-5,
            max_iteration=100
        )
        
        reg = o3d.pipelines.registration.registration_generalized_icp(
            source, target, max_corr_distance, trans_init,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria
        )
        
        translation = reg.transformation[:3, 3]
        fitness = reg.fitness
        inlier_rmse = reg.inlier_rmse
        
        # print(f"    Translation: dx={translation[0]:.3f}, dy={translation[1]:.3f}, dz={translation[2]:.3f}")
        print(f"\nGICP fitness={fitness:.3f}, rmse={inlier_rmse:.3f}\n{str('-')*30}")
        
        return translation, fitness, inlier_rmse

    
    def estimate_motion_icp_single(self, prev_points, curr_points, init_transformation=None, refine_rotation=False, icp_max_distance=VOXEL_SIZE*1.5):
        if prev_points is None or curr_points is None:
            print("cal_pose.py - estimate_motion_icp_single :  points are None")
            return np.eye(4), False, 0.0
        
        if len(prev_points) < 100 or len(curr_points) < 100:
            print("cal_pose.py - estimate_motion_icp_single :  points are not enough")
            return np.eye(4), False, 0.0
        
        prev_down = self.downsample_points(prev_points, voxel_size=VOXEL_SIZE)
        curr_down = self.downsample_points(curr_points, voxel_size=VOXEL_SIZE)
        
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(curr_down)
        target.points = o3d.utility.Vector3dVector(prev_down)
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS, max_nn=GICP_NN)
        )
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS, max_nn=GICP_NN)
        )
        
        source.orient_normals_consistent_tangent_plane(k=15)
        target.orient_normals_consistent_tangent_plane(k=15)
        
        trans_init = init_transformation if init_transformation is not None else np.eye(4, dtype=np.float64)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=100
        )
        max_corr_distance = icp_max_distance
        reg_p2l = o3d.pipelines.registration.registration_generalized_icp(
            source, target, max_corr_distance, trans_init,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria
        )
        
        transformation = reg_p2l.transformation
        fitness = reg_p2l.fitness
        inlier_rmse = reg_p2l.inlier_rmse
        
        R_p2l = transformation[:3, :3]
        rotation_angle_p2l = np.arccos(np.clip((np.trace(R_p2l) - 1) / 2, -1, 1))
        
        if fitness > 0.95 and inlier_rmse < VOXEL_SIZE*3:         ############################################################################THRESHOLDTHRESHOLDTHRESHOLDTHRESHOLDTHRESHOLDTHRESHOLDTHRESHOLDTHRESHOLDTHRESHOLDTHRESHOLDTHRESHOLDTHRESHOLD
            print(f"    [TRANSLATION MODE] fitness={fitness:.3f}, rot={np.degrees(rotation_angle_p2l):.2f}°")
            return transformation, True, fitness
        else:
            print(f"    [ROTATION MODE] fitness={fitness:.3f}")
            rotation_angle, conf_rotation = self.estimate_rotation_combined(prev_points, curr_points)
            
            if refine_rotation:
                rotation_angle, conf_rotation = self.estimate_rotation_combined(
                    prev_points, curr_points, 
                    refine=True, 
                    coarse_angle=rotation_angle
                )
            
            if conf_rotation < 0.2:
                print(f"    !!!!!!![REJECTED] Low rotation confidence: {conf_rotation:.3f}")
                return np.eye(4), False, 0.0
            
            translation, fitness, inlier_rmse = self.estimate_translation_icp(
                prev_points, curr_points, rotation_angle
            )
            
            if fitness < FITNESS or inlier_rmse > 0.5:
                print(f"    !!!!!!![REJECTED] fitness={fitness:.3f}, rmse={inlier_rmse:.3f}")
                return np.eye(4), False, 0.0
            
            transformation = np.eye(4)
            transformation[:3, :3] = np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1]
            ])
            transformation[:3, 3] = translation
            return transformation, True, fitness
    
    def estimate_motion_multiframe(self, point_cloud_buffer, pairs=None, 
                                   current_global_pos=None, current_global_yaw=None):
        n_frames = len(point_cloud_buffer)
        if n_frames < 2:
            return None
        
        pairwise_transformations = {}
        fitness_scores = {}
        sequential_pairs = [(i, i+1) for i in range(n_frames - 1)]
        
        for i, j in sequential_pairs:
            prev_trans = np.eye(4)
            
            trans, success, fitness = self.estimate_motion_icp_single(
                        point_cloud_buffer[i], 
                        point_cloud_buffer[j],
                        init_transformation=np.eye(4),
                        refine_rotation=True,
                        )
            
            if success:
                pairwise_transformations[(i, j)] = trans
                fitness_scores[(i, j)] = fitness
            else:
                print(f"    Frame {i}->{j}: FAILED")
                if i == n_frames-2 and j == n_frames -1:
                    return None
        
        for i, j in pairs:
            if (i, j) in pairwise_transformations or j != i + 1:
                if i >= n_frames or j >= n_frames:
                    continue
                
                if j > i + 1:
                    trans, success, fitness = self.estimate_motion_icp_single(
                        point_cloud_buffer[i], 
                        point_cloud_buffer[j],
                        init_transformation=np.eye(4),
                        refine_rotation=False,
                    )
                    
                    if success:
                        pairwise_transformations[(i, j)] = trans
                        fitness_scores[(i, j)] = fitness
                        print(f"    Frame {i}->{j}: fitness={fitness:.3f}")
                        
                        self._update_sequential_transforms(
                            pairwise_transformations, 
                            fitness_scores,
                            i, j, trans, fitness
                        )
                    else:
                        print(f"    Frame {i}->{j}: FAILED ")
        
        if len(pairwise_transformations) == 0:
            return None
        
        frame_positions = [np.array([0.0, 0.0])]
        frame_yaws = [0.0]
        
        for i in range(n_frames - 1):
            trans = pairwise_transformations.get((i, i+1), np.eye(4))
            
            local_x = trans[0, 3]
            local_y = trans[1, 3]
            R = trans[:3, :3].copy()
            
            rot = Rotation.from_matrix(R)
            euler_inv = rot.as_euler('ZYX', degrees=False)
            delta_yaw = euler_inv[0]
            
            print(f"    Frame {i}->{i+1}: dx={local_x:.2f}m, dy={local_y:.2f}m, yaw={np.degrees(delta_yaw):.2f}°")
            
            prev_yaw = frame_yaws[-1]
            cos_yaw = np.cos(prev_yaw)
            sin_yaw = np.sin(prev_yaw)
            
            global_x = frame_positions[-1][0] + cos_yaw * local_x - sin_yaw * local_y
            global_y = frame_positions[-1][1] + sin_yaw * local_x + cos_yaw * local_y
            
            frame_positions.append(np.array([global_x, global_y]))
            frame_yaws.append(normalize_angle(prev_yaw + delta_yaw))
        
        global_positions = []
        global_yaws = []
        
        if current_global_pos is not None and current_global_yaw is not None:
            for i in range(len(frame_positions)):
                cos_yaw = np.cos(current_global_yaw)
                sin_yaw = np.sin(current_global_yaw)
                
                global_pos = current_global_pos.copy()
                global_pos[0] += cos_yaw * frame_positions[i][0] - sin_yaw * frame_positions[i][1]
                global_pos[1] += sin_yaw * frame_positions[i][0] + cos_yaw * frame_positions[i][1]
                
                global_positions.append(global_pos)
                global_yaws.append(normalize_angle(current_global_yaw + frame_yaws[i]))
            
            return frame_positions, frame_yaws, global_positions, global_yaws
        
        return frame_positions, frame_yaws, None, None
    
    def _update_sequential_transforms(self, pairwise_transformations, fitness_scores, 
                                      start_idx, end_idx, skip_trans, skip_fitness):
        path_trans = np.eye(4)
        path_fitness_sum = 0.0
        path_count = 0
        
        for i in range(start_idx, end_idx):
            if (i, i+1) in pairwise_transformations:
                path_trans = path_trans @ pairwise_transformations[(i, i+1)]
                path_fitness_sum += fitness_scores.get((i, i+1), 0.0)
                path_count += 1
        
        if path_count == 0:
            return
        
        avg_path_fitness = path_fitness_sum / path_count
        
        if skip_fitness > avg_path_fitness + 0.1:
            print(f"    Correcting path {start_idx}->{end_idx}: "
                  f"path_fitness={avg_path_fitness:.3f} < skip_fitness={skip_fitness:.3f}")
            
            path_trans_inv = np.linalg.inv(path_trans)
            correction = skip_trans @ path_trans_inv
            
            n_steps = end_idx - start_idx
            
            corr_trans = correction[:3, 3]
            corr_R = correction[:3, :3]
            corr_rot = Rotation.from_matrix(corr_R)
            corr_euler = corr_rot.as_euler('ZYX', degrees=False)
            
            delta_trans_per_step = corr_trans / n_steps
            delta_yaw_per_step = corr_euler[0] / n_steps
            
            for i in range(start_idx, end_idx):
                if (i, i+1) in pairwise_transformations:
                    old_trans = pairwise_transformations[(i, i+1)].copy()
                    
                    old_trans[0, 3] += delta_trans_per_step[0]
                    old_trans[1, 3] += delta_trans_per_step[1]
                    old_trans[2, 3] += delta_trans_per_step[2]
                    
                    old_R = old_trans[:3, :3]
                    old_rot = Rotation.from_matrix(old_R)
                    old_euler = old_rot.as_euler('ZYX', degrees=False)
                    new_euler = old_euler.copy()
                    new_euler[0] += delta_yaw_per_step
                    
                    new_R = Rotation.from_euler('ZYX', new_euler, degrees=False).as_matrix()
                    old_trans[:3, :3] = new_R
                    
                    pairwise_transformations[(i, i+1)] = old_trans
                    
                    print(f"      Adjusted ({i},{i+1}): "
                          f"dt=({delta_trans_per_step[0]:.3f},{delta_trans_per_step[1]:.3f}), "
                          f"dyaw={np.degrees(delta_yaw_per_step):.2f}°")


class GlobalMapManager:
    def __init__(self):
        self.global_map = {}
    
    def remove_cell_from_map(self, local_cell_map, robot_pos, robot_yaw, cell_size):
        """
        Remove cells from global map
        local_cell_map: dict with (cell_x, cell_y) as keys
        """
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        
        for (local_cell_x, local_cell_y), value in local_cell_map.items():
            # Convert cell coordinates to local world coordinates (cell center)
            local_x = (local_cell_x + 0.5) * cell_size
            local_y = (local_cell_y + 0.5) * cell_size
            
            # Rotate to global frame
            rotated_x = cos_yaw * local_x - sin_yaw * local_y
            rotated_y = sin_yaw * local_x + cos_yaw * local_y
            
            # Transform to global position
            global_x = robot_pos[0] + rotated_x
            global_y = robot_pos[1] + rotated_y
            
            # Convert back to cell coordinates
            global_cell_x = int(np.floor(global_x / cell_size))
            global_cell_y = int(np.floor(global_y / cell_size))
            key = (global_cell_x, global_cell_y)
            
            self.global_map.pop(key, None)
    
    def add_to_global_map(self, local_cell_map, robot_pos, robot_yaw, cell_size):
        """
        Add cells to global map with proper coordinate transformation
        local_cell_map: dict with (cell_x, cell_y) as keys and 0/1/2 as values
        """
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        
        updates = {}
        for (local_cell_x, local_cell_y), value in local_cell_map.items():
            # Convert cell coordinates to local world coordinates (cell center)
            local_x = (local_cell_x + 0.5) * cell_size
            local_y = (local_cell_y + 0.5) * cell_size
            
            # Rotate to global frame
            rotated_x = cos_yaw * local_x - sin_yaw * local_y
            rotated_y = sin_yaw * local_x + cos_yaw * local_y
            
            # Transform to global position
            global_x = robot_pos[0] + rotated_x
            global_y = robot_pos[1] + rotated_y
            
            # Convert back to cell coordinates
            global_cell_x = int(np.floor(global_x / cell_size))
            global_cell_y = int(np.floor(global_y / cell_size))
            key = (global_cell_x, global_cell_y)
            
            # Update cell: obstacles (2) take priority over road (1)
            if key in self.global_map:
                updates[key] = max(self.global_map[key], value)
            else:
                updates[key] = value
        
        self.global_map.update(updates)
    
    def update_global_map_for_buffer(self, frame_positions, frame_yaws, 
                                     buffer_positions, buffer_yaws,
                                     local_cell_map_buffer, robot_pos, robot_yaw, cell_size):
        """
        Update global map when buffer frame positions are updated
        """
        print(f"  Updating global map for all {len(frame_positions)} frames in buffer")
        
        buffer_start_global_pos = robot_pos.copy()
        buffer_start_global_yaw = robot_yaw
        
        # First, remove all old cells
        for i in range(len(buffer_positions)):
            if i < len(local_cell_map_buffer):
                old_global_pos = buffer_start_global_pos.copy()
                old_global_yaw = buffer_start_global_yaw
                
                rel_pos = buffer_positions[i]
                rel_yaw = buffer_yaws[i]
                
                cos_start_yaw = np.cos(buffer_start_global_yaw)
                sin_start_yaw = np.sin(buffer_start_global_yaw)
                
                old_global_pos[0] += cos_start_yaw * rel_pos[0] - sin_start_yaw * rel_pos[1]
                old_global_pos[1] += sin_start_yaw * rel_pos[0] + cos_start_yaw * rel_pos[1]
                old_global_yaw = normalize_angle(buffer_start_global_yaw + rel_yaw)
                
                self.remove_cell_from_map(
                    local_cell_map_buffer[i],
                    old_global_pos,
                    old_global_yaw,
                    cell_size
                )
        
        # Then, add all cells with new positions
        for i in range(len(frame_positions)):
            if i < len(local_cell_map_buffer):
                new_global_pos = buffer_start_global_pos.copy()
                new_global_yaw = buffer_start_global_yaw
                
                rel_pos = frame_positions[i]
                rel_yaw = frame_yaws[i]
                
                cos_start_yaw = np.cos(buffer_start_global_yaw)
                sin_start_yaw = np.sin(buffer_start_global_yaw)
                
                new_global_pos[0] += cos_start_yaw * rel_pos[0] - sin_start_yaw * rel_pos[1]
                new_global_pos[1] += sin_start_yaw * rel_pos[0] + cos_start_yaw * rel_pos[1]
                new_global_yaw = normalize_angle(buffer_start_global_yaw + rel_yaw)
                
                self.add_to_global_map(
                    local_cell_map_buffer[i],
                    new_global_pos,
                    new_global_yaw,
                    cell_size
                )