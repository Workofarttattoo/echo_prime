"""
Enhanced Visual Abstraction Capabilities for ARC-AGI Tasks.

This module provides sophisticated pattern recognition and abstraction capabilities
specifically designed for Abstract Reasoning Corpus (ARC-AGI) tasks, including:
- Grid pattern recognition and abstraction
- Object detection and relationship analysis
- Transformation inference
- Symmetry and periodicity detection
- Color and shape pattern analysis
- Spatial reasoning and analogies
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, NamedTuple
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict, Counter
import cv2
from PIL import Image
import math


class PatternType(Enum):
    """Types of visual patterns"""
    GRID_STRUCTURE = "grid_structure"
    OBJECT_RELATIONSHIPS = "object_relationships"
    SYMMETRY = "symmetry"
    PERIODICITY = "periodicity"
    TRANSFORMATION = "transformation"
    COLOR_PATTERNS = "color_patterns"
    SHAPE_PATTERNS = "shape_patterns"
    SPATIAL_ARRANGEMENT = "spatial_arrangement"


@dataclass
class GridCell:
    """Represents a cell in a grid with position and content"""
    row: int
    col: int
    color: int
    shape: Optional[str] = None
    size: Optional[int] = None

    def __hash__(self):
        return hash((self.row, self.col, self.color, self.shape, self.size))


@dataclass
class VisualObject:
    """Represents a detected visual object"""
    cells: List[GridCell]
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    color: int
    shape_type: str
    symmetry: Dict[str, bool]
    size: int
    position: Tuple[int, int]  # centroid

    def __hash__(self):
        return hash(tuple(sorted(self.cells)))


@dataclass
class PatternMatch:
    """Represents a detected pattern"""
    pattern_type: PatternType
    confidence: float
    description: str
    parameters: Dict[str, Any]
    examples: List[Any]
    transformations: List[str]


class GridAnalyzer:
    """
    Analyzes grid-based visual patterns typical in ARC-AGI tasks.
    """

    def __init__(self):
        self.color_map = {}  # Maps color values to semantic meanings

    def analyze_grid(self, grid: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive analysis of a grid-based visual pattern.

        Args:
            grid: 2D numpy array representing the grid

        Returns:
            Dictionary with detailed grid analysis
        """
        if grid.ndim != 2:
            raise ValueError("Grid must be 2D")

        height, width = grid.shape

        analysis = {
            'dimensions': (height, width),
            'unique_colors': len(np.unique(grid)),
            'color_distribution': self._analyze_color_distribution(grid),
            'symmetries': self._detect_symmetries(grid),
            'periodicity': self._detect_periodicity(grid),
            'objects': self._detect_objects(grid),
            'spatial_patterns': self._analyze_spatial_patterns(grid),
            'transformations': self._infer_transformations(grid),
            'complexity_score': self._calculate_complexity(grid)
        }

        return analysis

    def _analyze_color_distribution(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of colors in the grid."""
        unique_colors, counts = np.unique(grid, return_counts=True)
        total_cells = grid.size

        distribution = {}
        for color, count in zip(unique_colors, counts):
            percentage = count / total_cells
            distribution[int(color)] = {
                'count': int(count),
                'percentage': percentage,
                'density': count / max(1, self._calculate_connected_components(grid, color))
            }

        # Identify background vs foreground colors
        sorted_colors = sorted(distribution.items(), key=lambda x: x[1]['percentage'], reverse=True)
        background_color = sorted_colors[0][0] if sorted_colors else 0

        return {
            'distribution': distribution,
            'background_color': background_color,
            'foreground_colors': [c for c, _ in sorted_colors[1:]],
            'color_entropy': self._calculate_color_entropy(counts, total_cells)
        }

    def _detect_symmetries(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect various types of symmetry in the grid."""
        symmetries = {}

        # Horizontal symmetry
        symmetries['horizontal'] = np.array_equal(grid, np.flipud(grid))

        # Vertical symmetry
        symmetries['vertical'] = np.array_equal(grid, np.fliplr(grid))

        # Diagonal symmetry (main diagonal)
        symmetries['diagonal_main'] = np.array_equal(grid, grid.T)

        # Diagonal symmetry (anti-diagonal)
        symmetries['diagonal_anti'] = np.array_equal(grid, np.flipud(grid).T)

        # Rotational symmetry (90, 180, 270 degrees)
        symmetries['rotational_90'] = np.array_equal(grid, np.rot90(grid, 1))
        symmetries['rotational_180'] = np.array_equal(grid, np.rot90(grid, 2))
        symmetries['rotational_270'] = np.array_equal(grid, np.rot90(grid, 3))

        # Partial symmetries (for non-square grids)
        symmetries['partial_horizontal'] = self._check_partial_symmetry(grid, 'horizontal')
        symmetries['partial_vertical'] = self._check_partial_symmetry(grid, 'vertical')

        return symmetries

    def _check_partial_symmetry(self, grid: np.ndarray, symmetry_type: str) -> float:
        """Check partial symmetry and return similarity score."""
        height, width = grid.shape

        if symmetry_type == 'horizontal' and height > 1:
            # Compare top half with bottom half
            mid = height // 2
            top = grid[:mid]
            bottom = grid[height-mid:] if height % 2 == 0 else grid[height-mid-1:]

            # Flip bottom to compare
            bottom_flipped = np.flipud(bottom)
            if bottom_flipped.shape[0] > top.shape[0]:
                bottom_flipped = bottom_flipped[:top.shape[0]]

            matches = np.sum(top == bottom_flipped)
            total = top.size
            return matches / total if total > 0 else 0.0

        elif symmetry_type == 'vertical' and width > 1:
            # Compare left half with right half
            mid = width // 2
            left = grid[:, :mid]
            right = grid[:, width-mid:] if width % 2 == 0 else grid[:, width-mid-1:]

            # Flip right to compare
            right_flipped = np.fliplr(right)
            if right_flipped.shape[1] > left.shape[1]:
                right_flipped = right_flipped[:, :left.shape[1]]

            matches = np.sum(left == right_flipped)
            total = left.size
            return matches / total if total > 0 else 0.0

        return 0.0

    def _detect_periodicity(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect periodic patterns in the grid."""
        height, width = grid.shape
        periodicity = {}

        # Check for row periodicity
        row_periods = self._find_periods(grid, axis=0)
        periodicity['rows'] = row_periods

        # Check for column periodicity
        col_periods = self._find_periods(grid, axis=1)
        periodicity['columns'] = col_periods

        # Check for 2D periodicity (block patterns)
        block_periods = self._find_2d_periods(grid)
        periodicity['blocks'] = block_periods

        return periodicity

    def _find_periods(self, grid: np.ndarray, axis: int, max_period: int = 10) -> List[Tuple[int, float]]:
        """Find periodic patterns along a specific axis."""
        if axis == 0:  # Rows
            sequences = [grid[i, :] for i in range(grid.shape[0])]
        else:  # Columns
            sequences = [grid[:, i] for i in range(grid.shape[1])]

        periods = []

        for period in range(1, min(max_period + 1, len(sequences) // 2 + 1)):
            matches = 0
            total_comparisons = 0

            for i in range(len(sequences) - period):
                if np.array_equal(sequences[i], sequences[i + period]):
                    matches += 1
                total_comparisons += 1

            if total_comparisons > 0:
                similarity = matches / total_comparisons
                if similarity > 0.8:  # High similarity threshold
                    periods.append((period, similarity))

        return periods

    def _find_2d_periods(self, grid: np.ndarray) -> List[Tuple[Tuple[int, int], float]]:
        """Find 2D periodic block patterns."""
        height, width = grid.shape
        block_periods = []

        # Try different block sizes
        for block_h in range(1, min(5, height // 2 + 1)):
            for block_w in range(1, min(5, width // 2 + 1)):
                if block_h * 2 > height or block_w * 2 > width:
                    continue

                # Extract first block
                block1 = grid[:block_h, :block_w]

                # Check if pattern repeats
                matches = 0
                total_blocks = 0

                for r in range(0, height - block_h + 1, block_h):
                    for c in range(0, width - block_w + 1, block_w):
                        block = grid[r:r+block_h, c:c+block_w]
                        if block.shape == block1.shape:
                            if np.array_equal(block, block1):
                                matches += 1
                            total_blocks += 1

                if total_blocks > 1:
                    similarity = matches / total_blocks
                    if similarity > 0.9:
                        block_periods.append(((block_h, block_w), similarity))

        return block_periods

    def _detect_objects(self, grid: np.ndarray) -> List[VisualObject]:
        """Detect distinct objects in the grid."""
        height, width = grid.shape
        visited = np.zeros((height, width), dtype=bool)
        objects = []

        # Color-based object detection (connected components)
        for color in np.unique(grid):
            if color == 0:  # Skip background
                continue

            color_mask = (grid == color)
            labeled_mask, num_features = self._connected_components(color_mask)

            for obj_id in range(1, num_features + 1):
                obj_mask = (labeled_mask == obj_id)
                if np.sum(obj_mask) < 2:  # Skip single cells
                    continue

                # Get object cells
                obj_cells = []
                rows, cols = np.where(obj_mask)
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()

                for r, c in zip(rows, cols):
                    obj_cells.append(GridCell(r, c, int(color)))

                # Calculate centroid
                centroid_row = int(np.mean(rows))
                centroid_col = int(np.mean(cols))

                # Determine shape type
                shape_type = self._classify_shape(obj_mask)

                # Check symmetries
                obj_symmetries = self._check_object_symmetries(obj_mask)

                visual_obj = VisualObject(
                    cells=obj_cells,
                    bounding_box=(min_row, min_col, max_row, max_col),
                    color=int(color),
                    shape_type=shape_type,
                    symmetry=obj_symmetries,
                    size=len(obj_cells),
                    position=(centroid_row, centroid_col)
                )

                objects.append(visual_obj)

        return objects

    def _connected_components(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Find connected components in binary mask."""
        # Simple 4-connectivity connected components
        height, width = binary_mask.shape
        labeled = np.zeros((height, width), dtype=int)
        label_counter = 1

        for i in range(height):
            for j in range(width):
                if binary_mask[i, j] and labeled[i, j] == 0:
                    # Start new component
                    self._flood_fill(binary_mask, labeled, i, j, label_counter)
                    label_counter += 1

        return labeled, label_counter - 1

    def _flood_fill(self, mask: np.ndarray, labeled: np.ndarray, row: int, col: int, label: int):
        """Flood fill algorithm for connected components."""
        stack = [(row, col)]
        labeled[row, col] = label

        while stack:
            r, c = stack.pop()

            # Check 4-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and
                    mask[nr, nc] and labeled[nr, nc] == 0):
                    labeled[nr, nc] = label
                    stack.append((nr, nc))

    def _classify_shape(self, mask: np.ndarray) -> str:
        """Classify the shape of an object."""
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return "empty"

        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1
        area = np.sum(mask)
        perimeter = self._calculate_perimeter(mask)

        # Calculate shape features
        aspect_ratio = width / height if height > 0 else 1
        compactness = (perimeter ** 2) / area if area > 0 else 0

        # Classify based on features
        if aspect_ratio > 2:
            return "line_horizontal"
        elif aspect_ratio < 0.5:
            return "line_vertical"
        elif compactness < 20:
            return "blob"
        elif 20 <= compactness < 30:
            return "square_like"
        elif compactness < 40:
            return "circle_like"
        else:
            return "irregular"

    def _calculate_perimeter(self, mask: np.ndarray) -> int:
        """Calculate perimeter of binary shape."""
        # Simple perimeter calculation using morphological operations
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        perimeter_mask = mask.astype(np.uint8) - eroded
        return np.sum(perimeter_mask)

    def _check_object_symmetries(self, mask: np.ndarray) -> Dict[str, bool]:
        """Check symmetries of an individual object."""
        # Extract bounding box
        rows, cols = np.where(mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        obj_region = mask[min_row:max_row+1, min_col:max_col+1]

        symmetries = {}
        symmetries['horizontal'] = np.array_equal(obj_region, np.flipud(obj_region))
        symmetries['vertical'] = np.array_equal(obj_region, np.fliplr(obj_region))

        # For small objects, check rotational symmetry
        if obj_region.shape[0] == obj_region.shape[1]:  # Square region
            symmetries['rotational_90'] = np.array_equal(obj_region, np.rot90(obj_region, 1))
            symmetries['rotational_180'] = np.array_equal(obj_region, np.rot90(obj_region, 2))

        return symmetries

    def _analyze_spatial_patterns(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial patterns and arrangements."""
        patterns = {}

        # Check for gradients
        patterns['gradients'] = self._detect_gradients(grid)

        # Check for clusters
        patterns['clusters'] = self._detect_clusters(grid)

        # Check for boundaries and edges
        patterns['edges'] = self._detect_edges(grid)

        # Check for fractal-like patterns
        patterns['fractal_dimension'] = self._estimate_fractal_dimension(grid)

        return patterns

    def _detect_gradients(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect color gradients in the grid."""
        gradients = []

        # Check horizontal gradients
        for row in range(grid.shape[0]):
            row_data = grid[row, :]
            if len(set(row_data)) > 1:  # Not uniform
                diff = np.diff(row_data.astype(float))
                if np.all(diff >= 0) or np.all(diff <= 0):  # Monotonic
                    gradients.append({
                        'type': 'horizontal',
                        'row': row,
                        'direction': 'increasing' if np.mean(diff) > 0 else 'decreasing',
                        'magnitude': np.mean(np.abs(diff))
                    })

        # Check vertical gradients
        for col in range(grid.shape[1]):
            col_data = grid[:, col]
            if len(set(col_data)) > 1:
                diff = np.diff(col_data.astype(float))
                if np.all(diff >= 0) or np.all(diff <= 0):
                    gradients.append({
                        'type': 'vertical',
                        'col': col,
                        'direction': 'increasing' if np.mean(diff) > 0 else 'decreasing',
                        'magnitude': np.mean(np.abs(diff))
                    })

        return gradients

    def _detect_clusters(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect color clusters in the grid."""
        clusters = []

        # For each color, find connected components
        for color in np.unique(grid):
            if color == 0:  # Skip background
                continue

            color_mask = (grid == color).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(color_mask)

            for label_id in range(1, num_labels):
                component_mask = (labels == label_id)
                area = np.sum(component_mask)

                if area > 1:  # Only consider clusters
                    rows, cols = np.where(component_mask)
                    centroid = (int(np.mean(rows)), int(np.mean(cols)))

                    clusters.append({
                        'color': int(color),
                        'area': int(area),
                        'centroid': centroid,
                        'bounding_box': (rows.min(), cols.min(), rows.max(), cols.max())
                    })

        return clusters

    def _detect_edges(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect edges and boundaries in the grid."""
        # Simple edge detection using difference operators
        edges = {}

        # Horizontal edges
        horizontal_edges = []
        for i in range(grid.shape[0] - 1):
            diff_row = grid[i, :] != grid[i + 1, :]
            if np.any(diff_row):
                edge_positions = np.where(diff_row)[0]
                horizontal_edges.append({
                    'row': i,
                    'positions': edge_positions.tolist()
                })

        # Vertical edges
        vertical_edges = []
        for j in range(grid.shape[1] - 1):
            diff_col = grid[:, j] != grid[:, j + 1]
            if np.any(diff_col):
                edge_positions = np.where(diff_col)[0]
                vertical_edges.append({
                    'col': j,
                    'positions': edge_positions.tolist()
                })

        edges['horizontal'] = horizontal_edges
        edges['vertical'] = vertical_edges
        edges['total_edges'] = len(horizontal_edges) + len(vertical_edges)

        return edges

    def _estimate_fractal_dimension(self, grid: np.ndarray) -> float:
        """Estimate fractal dimension using box counting method."""
        # Convert to binary for simplicity
        binary_grid = (grid > 0).astype(int)

        # Box counting for different box sizes
        sizes = [2, 4, 8, 16]
        counts = []

        for size in sizes:
            if size > min(binary_grid.shape):
                break

            count = 0
            for i in range(0, binary_grid.shape[0], size):
                for j in range(0, binary_grid.shape[1], size):
                    if np.any(binary_grid[i:i+size, j:j+size]):
                        count += 1
            counts.append(count)

        if len(counts) < 2:
            return 1.0  # Default dimension

        # Linear regression on log-log plot
        log_sizes = np.log(sizes[:len(counts)])
        log_counts = np.log(counts)

        # Simple slope calculation
        slope = np.polyfit(log_sizes, log_counts, 1)[0]
        dimension = -slope

        return max(1.0, min(2.0, dimension))  # Clamp to reasonable range

    def _infer_transformations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Infer possible transformations that could generate this grid."""
        transformations = []

        # Check for rotational transformations
        for k in [1, 2, 3]:  # 90, 180, 270 degrees
            rotated = np.rot90(grid, k)
            if np.array_equal(grid, rotated):
                transformations.append({
                    'type': 'rotation',
                    'degrees': k * 90,
                    'description': f'{k * 90} degree rotation symmetry'
                })

        # Check for reflection transformations
        if np.array_equal(grid, np.flipud(grid)):
            transformations.append({
                'type': 'reflection',
                'axis': 'horizontal',
                'description': 'Horizontal reflection symmetry'
            })

        if np.array_equal(grid, np.fliplr(grid)):
            transformations.append({
                'type': 'reflection',
                'axis': 'vertical',
                'description': 'Vertical reflection symmetry'
            })

        # Check for scaling patterns
        scales = self._detect_scaling_patterns(grid)
        transformations.extend(scales)

        return transformations

    def _detect_scaling_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect if grid follows scaling patterns."""
        patterns = []

        height, width = grid.shape

        # Check for self-similar patterns at different scales
        for scale in [2, 3, 4]:
            if height % scale == 0 and width % scale == 0:
                try:
                    # Downsample
                    small_grid = grid[::scale, ::scale]

                    # Upsample back using nearest neighbor
                    upsampled = np.kron(small_grid, np.ones((scale, scale), dtype=int))

                    # Check similarity
                    similarity = np.mean(grid == upsampled)
                    if similarity > 0.8:
                        patterns.append({
                            'type': 'scaling',
                            'scale_factor': scale,
                            'similarity': similarity,
                            'description': f'Self-similar pattern with {scale}x scaling'
                        })
                except:
                    continue

        return patterns

    def _calculate_connected_components(self, grid: np.ndarray, color: int) -> int:
        """Count connected components for a specific color."""
        color_mask = (grid == color)
        _, num_components = self._connected_components(color_mask)
        return num_components

    def _calculate_color_entropy(self, counts: np.ndarray, total: int) -> float:
        """Calculate Shannon entropy of color distribution."""
        probabilities = counts / total
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _calculate_complexity(self, grid: np.ndarray) -> float:
        """Calculate overall complexity score of the grid."""
        height, width = grid.shape
        total_cells = height * width

        # Factors contributing to complexity
        unique_colors = len(np.unique(grid))
        color_entropy = self._calculate_color_entropy(
            np.array([np.sum(grid == c) for c in np.unique(grid)]), total_cells
        )

        # Spatial complexity (based on edges)
        edges = self._detect_edges(grid)
        edge_density = edges['total_edges'] / max(1, (height - 1) + (width - 1))

        # Object complexity
        objects = self._detect_objects(grid)
        avg_object_size = np.mean([obj.size for obj in objects]) if objects else 0

        # Combine factors
        complexity = (
            0.3 * (unique_colors / 10) +  # Color diversity
            0.3 * (color_entropy / 4) +   # Color distribution complexity
            0.2 * edge_density +           # Edge complexity
            0.2 * min(1.0, len(objects) / 10)  # Object count complexity
        )

        return min(1.0, complexity)


class PatternTransformer:
    """
    Handles transformation inference and application for ARC-AGI tasks.
    """

    def __init__(self):
        self.grid_analyzer = GridAnalyzer()

    def infer_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """
        Infer the transformation that maps input_grid to output_grid.

        Args:
            input_grid: Input grid
            output_grid: Output grid after transformation

        Returns:
            Dictionary describing the inferred transformation
        """
        if input_grid.shape != output_grid.shape:
            return {'type': 'shape_change', 'description': 'Grids have different shapes'}

        transformations = []

        # Check for color transformations
        color_transform = self._infer_color_transformation(input_grid, output_grid)
        if color_transform:
            transformations.append(color_transform)

        # Check for spatial transformations
        spatial_transform = self._infer_spatial_transformation(input_grid, output_grid)
        if spatial_transform:
            transformations.append(spatial_transform)

        # Check for object-based transformations
        object_transform = self._infer_object_transformation(input_grid, output_grid)
        if object_transform:
            transformations.append(object_transform)

        # Determine most likely transformation
        if transformations:
            # Simple heuristic: prefer transformations with higher confidence
            best_transform = max(transformations, key=lambda x: x.get('confidence', 0))
            return best_transform

        return {'type': 'unknown', 'description': 'No clear transformation pattern detected'}

    def _infer_color_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Infer color-based transformations."""
        # Check if colors are shifted by constant amount
        diff_grid = output_grid.astype(int) - input_grid.astype(int)
        unique_diffs = np.unique(diff_grid)

        if len(unique_diffs) == 1 and unique_diffs[0] != 0:
            return {
                'type': 'color_shift',
                'shift_amount': int(unique_diffs[0]),
                'description': f'All colors shifted by {unique_diffs[0]}',
                'confidence': 0.9
            }

        # Check for color mapping
        input_colors = np.unique(input_grid)
        output_colors = np.unique(output_grid)

        if len(input_colors) == len(output_colors) and len(input_colors) > 1:
            # Try to find mapping
            color_map = {}
            consistent = True

            for in_color in input_colors:
                out_values = output_grid[input_grid == in_color]
                unique_out = np.unique(out_values)
                if len(unique_out) == 1:
                    color_map[int(in_color)] = int(unique_out[0])
                else:
                    consistent = False
                    break

            if consistent and len(color_map) > 0:
                return {
                    'type': 'color_mapping',
                    'mapping': color_map,
                    'description': f'Colors mapped according to {color_map}',
                    'confidence': 0.85
                }

        return None

    def _infer_spatial_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Infer spatial transformations like rotation, reflection, etc."""
        # Check rotations
        for k in [1, 2, 3]:
            rotated = np.rot90(input_grid, k)
            if np.array_equal(rotated, output_grid):
                return {
                    'type': 'rotation',
                    'degrees': k * 90,
                    'description': f'{k * 90} degree rotation',
                    'confidence': 1.0
                }

        # Check reflections
        if np.array_equal(np.flipud(input_grid), output_grid):
            return {
                'type': 'reflection',
                'axis': 'horizontal',
                'description': 'Horizontal reflection (flip up-down)',
                'confidence': 1.0
            }

        if np.array_equal(np.fliplr(input_grid), output_grid):
            return {
                'type': 'reflection',
                'axis': 'vertical',
                'description': 'Vertical reflection (flip left-right)',
                'confidence': 1.0
            }

        # Check for translation
        translation = self._detect_translation(input_grid, output_grid)
        if translation:
            return translation

        return None

    def _detect_translation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect if output is a translated version of input."""
        height, width = input_grid.shape

        # Try different translation offsets
        for dr in range(-height + 1, height):
            for dc in range(-width + 1, width):
                # Create translated version
                translated = np.zeros_like(input_grid)

                # Source region in input
                src_r_start = max(0, dr)
                src_r_end = min(height, height + dr)
                src_c_start = max(0, dc)
                src_c_end = min(width, width + dc)

                # Destination region in translated
                dst_r_start = max(0, -dr)
                dst_r_end = min(height, height - dr)
                dst_c_start = max(0, -dc)
                dst_c_end = min(width, width - dc)

                # Copy region
                translated[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
                    input_grid[src_r_start:src_r_end, src_c_start:src_c_end]

                # Check if matches output
                if np.array_equal(translated, output_grid):
                    return {
                        'type': 'translation',
                        'offset': (dr, dc),
                        'description': f'Translation by ({dr}, {dc})',
                        'confidence': 1.0
                    }

        return None

    def _infer_object_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Infer transformations based on object changes."""
        input_objects = self.grid_analyzer._detect_objects(input_grid)
        output_objects = self.grid_analyzer._detect_objects(output_grid)

        if not input_objects and not output_objects:
            return None

        # Check for object count changes
        if len(input_objects) != len(output_objects):
            return {
                'type': 'object_count_change',
                'input_count': len(input_objects),
                'output_count': len(output_objects),
                'description': f'Object count changed from {len(input_objects)} to {len(output_objects)}',
                'confidence': 0.7
            }

        # Check for size changes
        input_sizes = [obj.size for obj in input_objects]
        output_sizes = [obj.size for obj in output_objects]

        if input_sizes and output_sizes:
            size_ratio = np.mean(output_sizes) / np.mean(input_sizes) if np.mean(input_sizes) > 0 else 1

            if abs(size_ratio - 1) > 0.1:  # Significant size change
                return {
                    'type': 'object_scaling',
                    'scale_factor': size_ratio,
                    'description': f'Objects scaled by factor {size_ratio:.2f}',
                    'confidence': 0.8
                }

        return None

    def apply_transformation(self, grid: np.ndarray, transformation: Dict[str, Any]) -> np.ndarray:
        """Apply a transformation to a grid."""
        transform_type = transformation['type']

        if transform_type == 'rotation':
            degrees = transformation['degrees']
            k = degrees // 90
            return np.rot90(grid, k)

        elif transform_type == 'reflection':
            axis = transformation['axis']
            if axis == 'horizontal':
                return np.flipud(grid)
            elif axis == 'vertical':
                return np.fliplr(grid)

        elif transform_type == 'translation':
            dr, dc = transformation['offset']
            translated = np.zeros_like(grid)

            height, width = grid.shape
            src_r_start = max(0, dr)
            src_r_end = min(height, height + dr)
            src_c_start = max(0, dc)
            src_c_end = min(width, width + dc)

            dst_r_start = max(0, -dr)
            dst_r_end = min(height, height - dr)
            dst_c_start = max(0, -dc)
            dst_c_end = min(width, width - dc)

            translated[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
                grid[src_r_start:src_r_end, src_c_start:src_c_end]

            return translated

        elif transform_type == 'color_shift':
            shift = transformation['shift_amount']
            return np.clip(grid.astype(int) + shift, 0, 255).astype(grid.dtype)

        elif transform_type == 'color_mapping':
            mapping = transformation['mapping']
            result = grid.copy()
            for old_color, new_color in mapping.items():
                result[grid == old_color] = new_color
            return result

        return grid


class VisualAbstractionSystem:
    """
    Complete visual abstraction system for ARC-AGI tasks.
    """

    def __init__(self):
        self.grid_analyzer = GridAnalyzer()
        self.pattern_transformer = PatternTransformer()

    def analyze_task(self, input_grids: List[np.ndarray], output_grids: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze an ARC-AGI task with input-output pairs.

        Args:
            input_grids: List of input grids
            output_grids: List of corresponding output grids

        Returns:
            Dictionary with task analysis and solution hypothesis
        """
        if len(input_grids) != len(output_grids):
            raise ValueError("Input and output grids must have same length")

        # Analyze each input-output pair
        transformations = []
        for input_grid, output_grid in zip(input_grids, output_grids):
            transform = self.pattern_transformer.infer_transformation(input_grid, output_grid)
            transformations.append(transform)

        # Find common transformation pattern
        common_transform = self._find_common_transformation(transformations)

        # Analyze grid patterns
        input_patterns = [self.grid_analyzer.analyze_grid(grid) for grid in input_grids]
        output_patterns = [self.grid_analyzer.analyze_grid(grid) for grid in output_grids]

        # Generate solution hypothesis
        hypothesis = self._generate_solution_hypothesis(common_transform, input_patterns, output_patterns)

        return {
            'transformations': transformations,
            'common_transformation': common_transform,
            'input_patterns': input_patterns,
            'output_patterns': output_patterns,
            'solution_hypothesis': hypothesis,
            'confidence': self._calculate_solution_confidence(transformations, common_transform)
        }

    def _find_common_transformation(self, transformations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the most common transformation across examples."""
        if not transformations:
            return None

        # Count transformation types
        type_counts = Counter(t['type'] for t in transformations)

        # Find most common type
        most_common_type = type_counts.most_common(1)[0][0]

        # Check if all transformations are the same type
        if type_counts[most_common_type] == len(transformations):
            # For same type, find common parameters
            if most_common_type == 'rotation':
                degrees = [t.get('degrees', 0) for t in transformations]
                if len(set(degrees)) == 1:
                    return transformations[0]

            elif most_common_type == 'color_shift':
                shifts = [t.get('shift_amount', 0) for t in transformations]
                if len(set(shifts)) == 1:
                    return transformations[0]

            elif most_common_type == 'reflection':
                axes = [t.get('axis', '') for t in transformations]
                if len(set(axes)) == 1:
                    return transformations[0]

        return None

    def _generate_solution_hypothesis(self, common_transform: Optional[Dict[str, Any]],
                                    input_patterns: List[Dict[str, Any]],
                                    output_patterns: List[Dict[str, Any]]) -> str:
        """Generate a hypothesis about the solution."""
        if common_transform:
            transform_type = common_transform['type']
            description = common_transform['description']
            return f"The task involves applying a {transform_type} transformation: {description}"
        else:
            # Look for other patterns
            input_complexities = [p['complexity_score'] for p in input_patterns]
            output_complexities = [p['complexity_score'] for p in output_patterns]

            if np.mean(output_complexities) > np.mean(input_complexities):
                return "The task increases complexity, possibly by adding objects or patterns"
            elif np.mean(output_complexities) < np.mean(input_complexities):
                return "The task simplifies the pattern, possibly by removing elements"
            else:
                return "The task maintains similar complexity but changes the pattern structure"

    def _calculate_solution_confidence(self, transformations: List[Dict[str, Any]],
                                    common_transform: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in the solution hypothesis."""
        if not transformations:
            return 0.0

        if common_transform:
            # High confidence if we found a common transformation
            return 0.8
        else:
            # Lower confidence if transformations vary
            unique_types = len(set(t['type'] for t in transformations))
            return max(0.1, 1.0 - (unique_types - 1) * 0.2)

    def solve_task(self, input_grid: np.ndarray, task_analysis: Dict[str, Any], *args) -> np.ndarray:
        """
        Apply the learned transformation to solve a new input grid.

        Args:
            input_grid: New input grid to solve
            task_analysis: Analysis from analyze_task

        Returns:
            Predicted output grid
        """
        common_transform = task_analysis.get('common_transformation')

        if common_transform:
            return self.pattern_transformer.apply_transformation(input_grid, common_transform)
        else:
            # Fallback: try to find similar pattern in training examples
            # For now, return input unchanged
            return input_grid


# Export the main visual abstraction system
__all__ = ['VisualAbstractionSystem', 'GridAnalyzer', 'PatternTransformer', 'PatternMatch', 'PatternType']
