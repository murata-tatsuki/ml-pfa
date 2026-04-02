import numpy as np
import random

def minimum_enclosing_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    N次元の最小包囲球（Minimum Enclosing Sphere）を求める
    :param points: (M, N) の numpy 配列（M個のN次元点）
    :return: (球の中心, 半径)
    """
    def is_in_sphere(center: np.ndarray, radius: float, point: np.ndarray) -> bool:
        center = np.array(center)
        point = np.array(point)
        return np.linalg.norm(point - center) <= radius * (1 + 1e-14)
    
    def is_not_in_sphere_points(center: np.ndarray, radius: float, points: np.ndarray) -> np.ndarray:
        center = np.array(center)
        points = np.array(points)
        assert points.shape[1] == center.shape[0]
        distances = np.linalg.norm(points - center, axis=1)
        # return np.all(distances <= radius * (1 + 1e-14))
        return distances > radius * (1 + 1e-14)

    def get_sphere_from_points(boundary: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """ N+1個の点から最小包囲球を求める（線形代数を用いる） """
        np_boundary = np.array(boundary)
        if len(boundary) == 0:
            return np.zeros(points.shape[1]), 0.0
        if len(boundary) == 1:
            return boundary[0], 0.0
        if len(boundary) == 2:
            center = (np_boundary[0] + np_boundary[1]) / 2
            radius = np.linalg.norm(np_boundary[0] - center)
            return center, radius
        
        A = np.array([2 * (np_boundary[i] - np_boundary[0]) for i in range(1, len(boundary))])
        b = np.array([np.dot(np_boundary[i], np_boundary[i]) - np.dot(np_boundary[0], np_boundary[0]) for i in range(1, len(boundary))])
        try:
            center = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            center = np.mean(np_boundary, axis=0)
        radius = np.linalg.norm(np_boundary[0] - center)
        return center, radius

    def welzl_recursive(P: list[np.ndarray], R: list[np.ndarray], n: int) -> tuple[np.ndarray, float]:
        """ Welzl のアルゴリズム（再帰的に最小包囲球を求める） """
        if n == 0 or len(R) == points.shape[1] + 1:
            return get_sphere_from_points(R)
        idx = random.randint(0, n - 1)
        p = P[idx]
        P[idx], P[n - 1] = P[n - 1], P[idx]
        center, radius = welzl_recursive(P, R, n - 1)
        if is_in_sphere(center, radius, p):
            return center, radius
        return welzl_recursive(P, R + [p], n - 1)

    shuffled_points = points.tolist()
    random.shuffle(shuffled_points)
    center, radius = welzl_recursive(shuffled_points, [], len(shuffled_points))
    check_points = is_not_in_sphere_points(center, radius, points)
    while any(check_points):
        recheck_points = np.concatenate([points[check_points], points[~check_points]])
        recheck_points_list = recheck_points.tolist()
        center, radius = welzl_recursive(recheck_points_list, [], len(recheck_points_list))
        print("some points are not in the shere")
        check_points = is_not_in_sphere_points(center, radius, points)

    return welzl_recursive(shuffled_points, [], len(shuffled_points))

# 使用例（3次元の場合）
# points_3d = np.array([
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0],
#     [7.0, 8.0, 9.0],
#     [2.0, 8.0, 3.0],
#     [5.0, 3.0, 7.0]
# ])
# center, radius = minimum_enclosing_sphere(points_3d)
# print(f"最小包囲球の中心: {center}, 半径: {radius}")
