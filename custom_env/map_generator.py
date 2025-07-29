"""
Challenging road network map generator with complex paths and strategic routing
"""

import numpy as np
import random
from typing import List, Tuple, Dict


def create_road_network_map(grid_size: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
    """
    Create a challenging map with 5 destinations connected by complex road networks

    Features:
    - Multiple alternate routes
    - Dead ends and detours
    - Maze-like road patterns
    - Strategic bottlenecks
    - Randomized layouts

    Map legend:
    0 = obstacle/wall (impassable)
    1 = road (passable)
    2 = destination
    """
    height, width = grid_size

    # Initialize map with obstacles everywhere
    road_map = np.zeros(grid_size, dtype=int)

    # Define 5 destination positions with more spread
    destinations = {
        "hospital": (2, 2),  # Top-left corner
        "home": (2, width - 3),  # Top-right corner
        "market": (height // 2, width // 2),  # Center
        "bank": (height - 3, 2),  # Bottom-left corner
        "park": (height - 3, width - 3),  # Bottom-right corner
    }

    # Create complex road network
    road_positions = []

    # 1. Create main arterial roads (with gaps for complexity)
    road_positions.extend(_create_arterial_roads(destinations, height, width))

    # 2. Add maze-like secondary roads
    road_positions.extend(_create_maze_roads(destinations, height, width))

    # 3. Add dead ends and detours
    road_positions.extend(_create_dead_ends_and_detours(height, width))

    # 4. Create alternate routes between destinations
    road_positions.extend(_create_alternate_routes(destinations))

    # 5. Add random connecting roads for complexity
    road_positions.extend(_create_random_connections(height, width, num_connections=8))

    # Mark roads on map
    for pos in road_positions:
        if 0 <= pos[0] < height and 0 <= pos[1] < width:
            road_map[pos[0], pos[1]] = 1

    # Mark destinations on map
    for dest_pos in destinations.values():
        road_map[dest_pos[0], dest_pos[1]] = 2

    # Ensure destinations are connected to road network
    road_map = _ensure_destination_connectivity(road_map, destinations)

    # Add strategic complexity with controlled widening
    road_map = _add_strategic_complexity(road_map)

    return road_map, destinations, list(set(road_positions))


def _create_arterial_roads(
    destinations: Dict, height: int, width: int
) -> List[Tuple[int, int]]:
    """Create main arterial roads with strategic gaps"""
    roads = []

    # Main horizontal arterial (with gaps)
    main_row = height // 2
    for col in range(1, width - 1):
        # Create gaps to force detours
        if not (
            width // 3 <= col <= width // 3 + 2
            or 2 * width // 3 <= col <= 2 * width // 3 + 2
        ):
            roads.append((main_row, col))

    # Main vertical arterial (with gaps)
    main_col = width // 2
    for row in range(1, height - 1):
        # Create gaps to force detours
        if not (
            height // 3 <= row <= height // 3 + 2
            or 2 * height // 3 <= row <= 2 * height // 3 + 2
        ):
            roads.append((row, main_col))

    return roads


def _create_maze_roads(
    destinations: Dict, height: int, width: int
) -> List[Tuple[int, int]]:
    """Create maze-like road patterns around destinations"""
    roads = []

    # Create L-shaped and zigzag patterns
    for dest_name, (dest_row, dest_col) in destinations.items():
        # Create L-shaped access roads
        if dest_name == "hospital":
            # Complex path to hospital
            roads.extend([(dest_row, c) for c in range(dest_col, dest_col + 6)])
            roads.extend([(r, dest_col + 5) for r in range(dest_row, dest_row + 4)])
            roads.extend(
                [(dest_row + 3, c) for c in range(dest_col + 5, dest_col + 10)]
            )

        elif dest_name == "home":
            # Zigzag path to home
            roads.extend([(dest_row, c) for c in range(dest_col - 6, dest_col)])
            roads.extend([(r, dest_col - 6) for r in range(dest_row, dest_row + 3)])
            roads.extend([(dest_row + 2, c) for c in range(dest_col - 8, dest_col - 6)])

        elif dest_name == "bank":
            # Spiral approach to bank
            roads.extend([(dest_row, c) for c in range(dest_col, dest_col + 5)])
            roads.extend([(r, dest_col + 4) for r in range(dest_row - 4, dest_row)])
            roads.extend([(dest_row - 4, c) for c in range(dest_col + 4, dest_col + 8)])

        elif dest_name == "park":
            # Multi-level path to park
            roads.extend([(dest_row, c) for c in range(dest_col - 5, dest_col)])
            roads.extend([(r, dest_col - 5) for r in range(dest_row - 3, dest_row)])
            roads.extend([(dest_row - 3, c) for c in range(dest_col - 8, dest_col - 5)])

    return roads


def _create_dead_ends_and_detours(height: int, width: int) -> List[Tuple[int, int]]:
    """Create dead ends and detour roads to add complexity"""
    roads = []

    # Add dead ends in each quadrant
    quadrants = [
        (height // 4, width // 4),  # Top-left quadrant
        (height // 4, 3 * width // 4),  # Top-right quadrant
        (3 * height // 4, width // 4),  # Bottom-left quadrant
        (3 * height // 4, 3 * width // 4),  # Bottom-right quadrant
    ]

    for qr, qc in quadrants:
        # Create dead-end branches
        dead_end_length = random.randint(3, 6)
        direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])

        for i in range(dead_end_length):
            nr, nc = qr + direction[0] * i, qc + direction[1] * i
            if 0 < nr < height - 1 and 0 < nc < width - 1:
                roads.append((nr, nc))

    # Add detour loops
    center_r, center_c = height // 2, width // 2

    # Small detour loop in upper area
    loop_roads = [
        (center_r - 6, center_c - 2),
        (center_r - 6, center_c - 1),
        (center_r - 6, center_c),
        (center_r - 5, center_c),
        (center_r - 4, center_c),
        (center_r - 4, center_c - 1),
        (center_r - 4, center_c - 2),
    ]
    roads.extend(
        [(r, c) for r, c in loop_roads if 0 < r < height - 1 and 0 < c < width - 1]
    )

    # Small detour loop in lower area
    loop_roads = [
        (center_r + 4, center_c + 1),
        (center_r + 4, center_c + 2),
        (center_r + 4, center_c + 3),
        (center_r + 5, center_c + 3),
        (center_r + 6, center_c + 3),
        (center_r + 6, center_c + 2),
        (center_r + 6, center_c + 1),
    ]
    roads.extend(
        [(r, c) for r, c in loop_roads if 0 < r < height - 1 and 0 < c < width - 1]
    )

    return roads


def _create_alternate_routes(destinations: Dict) -> List[Tuple[int, int]]:
    """Create multiple alternate routes between key destinations"""
    roads = []

    # Create indirect routes that are longer but avoid main roads
    hospital = destinations["hospital"]
    market = destinations["market"]
    park = destinations["park"]
    home = destinations["home"]
    bank = destinations["bank"]

    # Alternate route: Hospital -> Park (via north detour)
    roads.extend(_create_detour_path(hospital, park, "north"))

    # Alternate route: Home -> Bank (via south detour)
    roads.extend(_create_detour_path(home, bank, "south"))

    # Alternate route: Market -> Hospital (via west detour)
    roads.extend(_create_detour_path(market, hospital, "west"))

    return roads


def _create_detour_path(
    start: Tuple[int, int], end: Tuple[int, int], detour_direction: str
) -> List[Tuple[int, int]]:
    """Create a detour path between two points"""
    roads = []
    sr, sc = start
    er, ec = end

    if detour_direction == "north":
        # Go up first, then across, then down
        mid_r = min(sr, er) - 3
        roads.extend([(r, sc) for r in range(sr, mid_r, -1) if r > 0])
        roads.extend([(mid_r, c) for c in range(sc, ec, 1 if ec > sc else -1)])
        roads.extend([(r, ec) for r in range(mid_r, er)])

    elif detour_direction == "south":
        # Go down first, then across, then up
        mid_r = max(sr, er) + 3
        roads.extend([(r, sc) for r in range(sr, mid_r)])
        roads.extend([(mid_r, c) for c in range(sc, ec, 1 if ec > sc else -1)])
        roads.extend([(r, ec) for r in range(mid_r, er, -1) if r > 0])

    elif detour_direction == "west":
        # Go left first, then down/up, then right
        mid_c = min(sc, ec) - 3
        roads.extend([(sr, c) for c in range(sc, mid_c, -1) if c > 0])
        roads.extend([(r, mid_c) for r in range(sr, er, 1 if er > sr else -1)])
        roads.extend([(er, c) for c in range(mid_c, ec)])

    return [(r, c) for r, c in roads if r > 0 and c > 0]


def _create_random_connections(
    height: int, width: int, num_connections: int
) -> List[Tuple[int, int]]:
    """Add random connecting roads for additional complexity"""
    roads = []

    for _ in range(num_connections):
        # Random starting point
        start_r = random.randint(2, height - 3)
        start_c = random.randint(2, width - 3)

        # Random direction and length
        direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1)])
        length = random.randint(3, 8)

        for i in range(length):
            nr = start_r + direction[0] * i
            nc = start_c + direction[1] * i
            if 1 < nr < height - 2 and 1 < nc < width - 2:
                roads.append((nr, nc))

    return roads


def _ensure_destination_connectivity(
    road_map: np.ndarray, destinations: Dict
) -> np.ndarray:
    """Ensure all destinations are connected to the road network"""
    for dest_pos in destinations.values():
        dr, dc = dest_pos
        # Add connecting roads around each destination
        for r_offset, c_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = dr + r_offset, dc + c_offset
            if 0 < nr < road_map.shape[0] - 1 and 0 < nc < road_map.shape[1] - 1:
                road_map[nr, nc] = 1

    return road_map


def _add_strategic_complexity(road_map: np.ndarray) -> np.ndarray:
    """Add strategic complexity with controlled road widening and bottlenecks"""
    height, width = road_map.shape
    enhanced_map = road_map.copy()

    # Find road positions
    road_positions = np.where(road_map == 1)

    # Selectively widen some roads (create highways vs narrow streets)
    for i in range(len(road_positions[0])):
        row, col = road_positions[0][i], road_positions[1][i]

        # Only widen certain roads (not all) to create strategic variety
        if random.random() < 0.15:  # Only 15% chance to widen (down from 30%)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if (
                    0 < nr < height - 1
                    and 0 < nc < width - 1
                    and enhanced_map[nr, nc] == 0
                ):  # Only convert obstacles to roads
                    enhanced_map[nr, nc] = 1

    return enhanced_map


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
