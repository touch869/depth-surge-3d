"""
Stereo pair generation module.

Converts frames and depth maps into stereo (left/right) pairs using disparity mapping.
"""

from __future__ import annotations

import cv2
import multiprocessing as mp
import numpy as np
import traceback
from pathlib import Path
from typing import Any

from ...utils import (
    depth_to_disparity,
    create_shifted_image,
    hole_fill_image,
)
from ...core.constants import PREVIEW_FRAME_SAMPLE_RATE


def _process_single_stereo_pair(
    args: tuple[np.ndarray, np.ndarray, str, str | None, str | None, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    PURE worker function to process a single stereo pair in parallel.

    Args:
        args: Tuple of (frame, depth_map, frame_name, left_path, right_path, settings)

    Returns:
        Tuple of (left_img, right_img, frame_name)

    Side effects:
        - Writes to left_path and right_path if provided
    """
    frame, depth_map, frame_name, left_path, right_path, settings = args

    # Create stereo pair
    disparity_map = depth_to_disparity(depth_map, settings["baseline"], settings["focal_length"])

    left_img = create_shifted_image(frame, disparity_map, "left")
    right_img = create_shifted_image(frame, disparity_map, "right")

    # Apply hole filling
    if settings["hole_fill_quality"] in ["fast", "advanced"]:
        left_img = hole_fill_image(left_img, method=settings["hole_fill_quality"])
        right_img = hole_fill_image(right_img, method=settings["hole_fill_quality"])

    # Save if paths provided
    if left_path:
        cv2.imwrite(left_path, left_img)
    if right_path:
        cv2.imwrite(right_path, right_img)

    return left_img, right_img, frame_name


class StereoPairGenerator:
    """
    Generates stereo pairs from frames and depth maps.

    Responsibilities:
    - Convert depth maps to disparity maps
    - Create left/right shifted images
    - Apply hole filling
    - Parallel processing orchestration
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize stereo pair generator.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def create_stereo_pairs(
        self,
        frame_files: list[Path],
        depth_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Generate stereo pairs (streaming: reads each frame + depth map from disk lazily).

        Args:
            frame_files: List of frame file PATHS (read one at a time — low memory)
            depth_files: List of depth map file PATHS (read one at a time)
            directories: Dictionary of processing directories
            settings: Processing settings with baseline, focal_length, etc.
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Writes stereo pair images to disk
        """
        try:
            num_frames = len(frame_files)
            print(f"  Streaming stereo generation over {num_frames} frames (lazy disk read)...")

            results = []
            for i, (frame_file, depth_file) in enumerate(zip(frame_files, depth_files)):
                # Read ONE frame + ONE depth map from disk (auto-freed each iteration)
                frame = cv2.imread(str(frame_file))
                # Depth maps are saved as uint8 (depth*255); restore float [0,1]
                # range that depth_to_disparity expects.
                depth_uint = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                if frame is None or depth_uint is None:
                    print(f"Warning: could not read frame {i} or its depth map, skipping")
                    results.append(None)
                    continue
                depth_map = depth_uint.astype(np.float32) / 255.0
                del depth_uint

                frame_name = frame_file.stem

                # Determine save paths
                left_path = (
                    str(directories["left_frames"] / f"{frame_name}.png")
                    if settings.get("keep_intermediates", True) and "left_frames" in directories
                    else None
                )
                right_path = (
                    str(directories["right_frames"] / f"{frame_name}.png")
                    if settings.get("keep_intermediates", True) and "right_frames" in directories
                    else None
                )

                result = _process_single_stereo_pair(
                    (frame, depth_map, frame_name, left_path, right_path, settings)
                )
                results.append(result)

                # Free this frame's memory before reading the next
                del frame, depth_map

                # Update progress
                if progress_tracker is not None and (i % 5 == 0 or i == num_frames - 1):
                    progress_tracker.update_progress(
                        "Creating stereo pairs",
                        phase="stereo_generation",
                        frame_num=i + 1,
                        step_name="Stereo Pair Creation",
                        step_progress=i + 1,
                        step_total=num_frames,
                    )

                # Send preview frame for left eye
                if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
                    if i % PREVIEW_FRAME_SAMPLE_RATE == 0 or i == num_frames - 1:
                        if left_path:
                            progress_tracker.send_preview_frame(
                                Path(left_path), "stereo_left", i + 1
                            )

            return True

        except Exception as e:
            print(f"Error creating stereo pairs: {e}")
            traceback.print_exc()
            return False
