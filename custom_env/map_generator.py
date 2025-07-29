"""
Road network map generator with connected destinations
"""

import numpy as np
from typing import List, Tuple, Dict


def create_road_network_map(grid_size: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
    """
    Create a map with 5 destinations connected by roads and obstacles

    Map legend:
    0 = obstacle/wall (impassable)
    1 = road (passable)
    2 = destination
    """
    height, width = grid_size

    # Initialize map with obstacles everywhere
    road_map = np.zeros(grid_size, dtype=int)

    # Define 5 destination positions (corners and center)
    destinations = {
        "hospital": (1, 1),  # Top-left
        "home": (1, width - 2),  # Top-right
        "market": (height // 2, width // 2),  # Center
        "bank": (height - 2, 1),  # Bottom-left
        "park": (height - 2, width - 2),  # Bottom-right
    }

    # Create road network connecting all destinations
    road_positions = []

    # Create main roads connecting destinations
    road_positions.extend(
        _create_path(destinations["hospital"], destinations["home"])
    )  # hospital to home (top)
    road_positions.extend(
        _create_path(destinations["hospital"], destinations["bank"])
    )  # hospital to bank (left side)
    road_positions.extend(
        _create_path(destinations["home"], destinations["park"])
    )  # home to park (right side)
    road_positions.extend(
        _create_path(destinations["bank"], destinations["park"])
    )  # bank to park (bottom)
    road_positions.extend(
        _create_path(destinations["hospital"], destinations["market"])
    )  # hospital to market (diagonal)
    road_positions.extend(
        _create_path(destinations["home"], destinations["market"])
    )  # home to market (diagonal)
    road_positions.extend(
        _create_path(destinations["market"], destinations["bank"])
    )  # market to bank (diagonal)
    road_positions.extend(
        _create_path(destinations["market"], destinations["park"])
    )  # market to park (diagonal)

    # Mark roads on map
    for pos in road_positions:
        if 0 <= pos[0] < height and 0 <= pos[1] < width:
            road_map[pos[0], pos[1]] = 1

    # Mark destinations on map
    for dest_pos in destinations.values():
        road_map[dest_pos[0], dest_pos[1]] = 2

    # Add some road widening for better navigation
    road_map = _widen_roads(road_map)

    return road_map, destinations, list(set(road_positions))


def _create_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Create a path between two points using Manhattan distance"""
    path = []
    current = list(start)
    target = list(end)

    # Move horizontally first
    while current[1] != target[1]:
        path.append(tuple(current))
        if current[1] < target[1]:
            current[1] += 1
        else:
            current[1] -= 1

    # Then move vertically
    while current[0] != target[0]:
        path.append(tuple(current))
        if current[0] < target[0]:
            current[0] += 1
        else:
            current[0] -= 1

    path.append(tuple(current))  # Add final position
    return path


def _widen_roads(road_map: np.ndarray) -> np.ndarray:
    """Widen roads slightly for better navigation"""
    height, width = road_map.shape
    widened_map = road_map.copy()

    # Find all road positions
    road_positions = np.where(road_map == 1)

    # Add adjacent positions as roads (but not over destinations)
    for i in range(len(road_positions[0])):
        row, col = road_positions[0][i], road_positions[1][i]

        # Check 4-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < height and 0 <= nc < width and widened_map[nr, nc] == 0
            ):  # Only convert obstacles to roads
                # Add road with some probability to avoid making everything roads
                if np.random.random() < 0.3:  # 30% chance to widen
                    widened_map[nr, nc] = 1

    return widened_map


def get_valid_moves(
    road_map: np.ndarray, position: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Get valid moves from current position (only on roads or destinations)"""
    row, col = position
    valid_moves = []

    # Check 4-connected directions
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # up, down, left, right
        nr, nc = row + dr, col + dc
        if (
            0 <= nr < road_map.shape[0]
            and 0 <= nc < road_map.shape[1]
            and road_map[nr, nc] in [1, 2]
        ):  # Road or destination
            valid_moves.append((nr, nc))

    return valid_moves


def is_valid_position(road_map: np.ndarray, position: Tuple[int, int]) -> bool:
    """Check if position is valid (on road or destination)"""
    row, col = position
    if 0 <= row < road_map.shape[0] and 0 <= col < road_map.shape[1]:
        return road_map[row, col] in [1, 2]  # Road or destination
    return False
