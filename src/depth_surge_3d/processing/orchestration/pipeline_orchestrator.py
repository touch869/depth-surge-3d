"""
Processing pipeline orchestrator.

Coordinates the complete video processing pipeline across all specialized processors.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...io.operations import (
    create_output_directories,
    save_processing_settings,
    update_processing_status,
)
from ...utils.path_utils import generate_output_filename
from ...utils import (
    step_complete,
    saved_to,
    title_bar,
    completion_banner,
)
from ...utils.imaging.image_processing import (
    depth_to_disparity,
    create_shifted_image,
    resize_image,
    create_vr_frame,
    apply_center_crop,
)


class NullProgressTracker:
    """No-op progress tracker for CLI mode (no UI)."""
    def update_progress(self, *args, **kwargs):
        pass
    def send_preview_frame(self, *args, **kwargs):
        pass
    def send_preview_frame_from_array(self, *args, **kwargs):
        pass
    def finish(self, *args, **kwargs):
        pass


class ProcessingOrchestrator:
    """
    High-level pipeline control and step sequencing.

    Responsibilities:
    - Pipeline execution flow
    - Step sequencing
    - Progress tracking coordination
    - Error handling
    - Settings management
    - Console output formatting
    """

    def __init__(
        self,
        depth_processor,
        stereo_generator,
        distortion_processor,
        upscaler,
        vr_assembler,
        video_encoder,
        verbose: bool = False,
    ):
        """
        Initialize processing orchestrator.

        Args:
            depth_processor: DepthMapProcessor instance
            stereo_generator: StereoPairGenerator instance
            distortion_processor: DistortionProcessor instance
            upscaler: FrameUpscalerProcessor instance
            vr_assembler: VRFrameAssembler instance
            video_encoder: VideoEncoder instance
            verbose: Enable verbose output
        """
        self.depth_processor = depth_processor
        self.stereo_generator = stereo_generator
        self.distortion_processor = distortion_processor
        self.upscaler = upscaler
        self.vr_assembler = vr_assembler
        self.video_encoder = video_encoder
        self.verbose = verbose
        self._settings_file: Path | None = None  # Track settings file for error handling
        self._total_steps = 7  # Updated dynamically based on settings
        self._start_time: float = 0.0  # Track processing start time

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Main processing pipeline entry point.

        Args:
            video_path: Input video path
            output_dir: Output directory
            video_properties: Video metadata (frame_count, fps, etc.)
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Console output
            - Filesystem state management
            - Delegates to all processor modules
        """
        # Substitute a no-op tracker in CLI mode so downstream code never sees None
        if progress_tracker is None:
            progress_tracker = NullProgressTracker()
        try:
            # Start timer
            self._start_time = time.time()

            # Setup processing environment
            output_path, directories, self._settings_file = self._setup_processing(
                str(video_path), str(output_dir), settings, video_properties
            )

            # Execute processing pipeline
            success = self._execute_pipeline(
                str(video_path),
                output_path,
                directories,
                video_properties,
                settings,
                progress_tracker,
            )

            return success

        except Exception as e:
            print(f"Error in video processing: {e}")
            if self._settings_file:
                update_processing_status(self._settings_file, "failed", {"error": str(e)})
            return False

    def _execute_pipeline(
        self,
        video_path: str,
        output_path: Path,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Execute complete 8-step pipeline.

        Args:
            video_path: Input video path
            output_path: Output directory path
            directories: Dictionary of processing directories
            video_properties: Video metadata (frame_count, fps, etc.)
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            True if all steps successful, False otherwise

        Side effects:
            - Executes all pipeline steps
            - Progress updates
            - Console output
        """
        # Calculate total steps based on settings
        self._total_steps = self._get_total_steps(settings)

        # Step 1: Extract frames (delegated to video_encoder)
        frame_files = self.video_encoder.extract_frames(
            video_path, directories, video_properties, settings
        )
        if not frame_files:
            return False
        print(step_complete(f"Step 1: Extracted {len(frame_files)} frames"))
        self._print_saved_to(directories.get("frames"), "Extracted frames")
        print()  # Blank line after step

        fps = video_properties.get("fps", 30.0)

        # Step 2: Generate depth maps (delegated to depth_processor)
        # Returns depth map FILE PATHS (lazy-read by renderer) — NOT loaded arrays,
        # to avoid OOM on long 4K video.
        depth_files = self.depth_processor.generate_depth_maps(
            frame_files, settings, directories, progress_tracker
        )
        if depth_files is None:
            return False
        print(step_complete(f"Step 2: Generated {len(depth_files)} depth maps"))
        self._print_saved_to(directories.get("depth_maps"), "Depth maps")
        print()  # Blank line after step

        # NOTE: do NOT preload all frames into memory (OOM on long 4K video).
        # Renderer reads each frame + depth map from disk lazily (streaming).

        # Execute steps 3-8
        return self._execute_remaining_steps(
            directories,
            settings,
            frame_files,
            depth_files,
            fps,
            video_path,
            output_path,
            progress_tracker,
        )

    def _render_streaming(
        self,
        frame_files: list[Path],
        depth_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        fps: float,
        video_path: str,
        output_path: Path,
        progress_tracker=None,
    ) -> bool:
        """
        Single-frame streaming render: fuses steps 3→7 (stereo, crop, VR assembly)
        into one per-frame loop that holds ONE frame in memory and writes ONLY the
        final VR frame to disk.

        Valid only when apply_distortion=False and upscale_model='none' (our
        production default). With crop_factor=1.0 (default), center crop is identity.

        Peak memory: 1 frame + 1 depth map (~hundreds of MB) regardless of video
        length — avoids the OOM of loading all frames, and writes 1 encrypted frame
        through gocryptfs per iteration (vs ~6 in the phase-based pipeline).
        """
        from ...utils.path_utils import generate_output_filename

        num_frames = len(frame_files)
        baseline = settings["baseline"]
        focal_length = settings["focal_length"]
        per_eye_w = settings["per_eye_width"]
        per_eye_h = settings["per_eye_height"]
        vr_format = settings["vr_format"]
        crop_factor = max(0.5, min(1.0, float(settings.get("crop_factor", 1.0))))

        vr_dir = directories["vr_frames"]
        vr_dir.mkdir(parents=True, exist_ok=True)

        print(title_bar("Streaming render (single-frame pipeline: stereo→crop→VR)"))
        print(f"  {num_frames} frames, per_eye={per_eye_w}x{per_eye_h}, crop_factor={crop_factor}")
        t0 = time.time()

        for i, (frame_file, depth_file) in enumerate(zip(frame_files, depth_files)):
            frame_name = frame_file.stem

            # --- read 1 frame + 1 depth map (auto-freed each iteration) ---
            frame = cv2.imread(str(frame_file))
            depth_uint = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            if frame is None or depth_uint is None:
                print(f"Warning: could not read frame {i} or depth, skipping")
                continue
            depth_map = depth_uint.astype(np.float32) / 255.0
            del depth_uint

            # --- step 3: stereo (depth → disparity → shift left/right) ---
            disparity = depth_to_disparity(depth_map, baseline, focal_length)
            left_img = create_shifted_image(frame, disparity, "left")
            right_img = create_shifted_image(frame, disparity, "right")
            del frame, depth_map, disparity

            # --- step 5: center crop (identity when crop_factor=1.0) ---
            if crop_factor < 1.0:
                left_img = apply_center_crop(left_img, crop_factor)
                right_img = apply_center_crop(right_img, crop_factor)

            # --- step 7: VR assembly (resize to per_eye + hstack) ---
            left_final = resize_image(left_img, per_eye_w, per_eye_h)
            right_final = resize_image(right_img, per_eye_w, per_eye_h)
            vr_frame = create_vr_frame(left_final, right_final, vr_format)
            del left_img, right_img, left_final, right_final

            # --- write ONLY the VR frame (1 encrypted write through gocryptfs) ---
            cv2.imwrite(str(vr_dir / f"{frame_name}.png"), vr_frame)
            del vr_frame

            if progress_tracker is not None and (i % 10 == 0 or i == num_frames - 1):
                progress_tracker.update_progress(
                    f"Streaming render {i + 1}/{num_frames}",
                    phase="streaming_render",
                    frame_num=i + 1,
                    step_name="Streaming Render",
                    step_progress=i + 1,
                    step_total=num_frames,
                )
            if i % 200 == 0 or i == num_frames - 1:
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                eta = (num_frames - i - 1) / rate if rate > 0 else 0
                print(f"  [{i + 1}/{num_frames}] {rate:.1f} fps, ETA {eta/60:.0f}min")

        print(step_complete(f"Streaming render: {num_frames} VR frames"))

        # --- step 8: encode final video (unchanged) ---
        vr_frames_dir = directories.get("vr_frames")
        if not vr_frames_dir:
            return self._handle_step_error("VR frames directory not found")
        success = self.video_encoder.create_video(
            vr_frames_dir, directories["base"], video_path, settings
        )
        if success:
            output_filename = generate_output_filename(
                Path(video_path).name, settings["vr_format"], settings["vr_resolution"]
            )
            print(step_complete("Step 8: Created final video"))
            self._print_saved_to(directories["base"], f"Final output: {output_filename}")
        return success

    def _execute_remaining_steps(  # noqa: C901
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        frame_files: list[Path],
        depth_files: list[Path],
        fps: float,
        video_path: str,
        output_path: Path,
        progress_tracker=None,
        current_step: int = 3,
    ) -> bool:
        """
        Execute remaining pipeline steps after depth map generation.
        Args:
            directories: Dictionary of processing directories
            settings: Processing settings
            frame_files: List of extracted frame file PATHS (read lazily)
            depth_files: List of depth map file PATHS (read lazily)
            fps: Video frames per second
            video_path: Input video path
            output_path: Output directory path
            progress_tracker: Optional progress tracker
            current_step: Current step number

        Returns:
            True if successful, False otherwise

        Side effects:
            - Executes steps 3-8 of pipeline
            - Progress updates
        """
        num_frames = len(frame_files)

        # ── FAST PATH: single-frame streaming (steps 3→7 fused) ──
        # When no fisheye distortion and no AI upscaling (our production default:
        # --no-distortion, crop_factor=1.0, no upscale), the per-frame chain is
        # stereo → (crop is identity) → resize → hstack → write VR frame.
        # Fusing into one loop writes ONLY the VR frame (1 encrypted write/frame
        # through gocryptfs instead of ~6), and holds 1 frame in memory (no OOM).
        # Falls back to the phase-based pipeline below if distortion/upscale enabled.
        if (
            not settings.get("apply_distortion", True)
            and settings.get("upscale_model", "none") == "none"
            and "vr_frames" in directories
        ):
            return self._render_streaming(
                frame_files, depth_files, directories, settings,
                fps, video_path, output_path, progress_tracker,
            )

        # Step 3: Create stereo pairs (delegated to stereo_generator)
        # Reads frames + depth maps from disk lazily (streaming, low memory).
        if not self.stereo_generator.create_stereo_pairs(
            frame_files, depth_files, directories, settings, progress_tracker
        ):
            return self._handle_step_error("Stereo pair creation failed")
        print(step_complete(f"Step 3: Created {num_frames} stereo pairs"))
        self._print_saved_to(directories.get("left_frames"), "Left frames")
        self._print_saved_to(directories.get("right_frames"), "Right frames")
        print()  # Blank line after left/right pair
        current_step += 1

        # Step 4: Apply fisheye distortion (optional - delegated to distortion_processor)
        if settings.get("apply_distortion", True):
            # Get left/right frame files
            left_dir = directories.get("left_frames")
            right_dir = directories.get("right_frames")
            if left_dir and right_dir:
                left_files = sorted(left_dir.glob("*.png"))
                right_files = sorted(right_dir.glob("*.png"))

                if left_files and right_files:
                    if not self.distortion_processor.apply_distortion(
                        left_files, right_files, directories, settings, progress_tracker
                    ):
                        return self._handle_step_error("Distortion failed")
                    print(
                        step_complete(
                            f"Step 4: Applied {settings['fisheye_projection']} fisheye distortion"
                        )
                    )
                    self._print_saved_to(directories.get("left_distorted"), "Distorted left frames")
                    self._print_saved_to(
                        directories.get("right_distorted"), "Distorted right frames"
                    )
                    print()  # Blank line after left/right pair
            current_step += 1

        # Step 5: Crop frames (delegated to distortion_processor)
        if not self.distortion_processor.crop_frames(
            directories, settings, progress_tracker, num_frames
        ):
            return self._handle_step_error("Frame cropping failed")
        print(
            step_complete(
                f"Step 5: Cropped {num_frames} frames to {settings['per_eye_width']}x{settings['per_eye_height']}"
            )
        )
        self._print_saved_to(directories.get("left_cropped"), "Cropped left frames")
        self._print_saved_to(directories.get("right_cropped"), "Cropped right frames")
        print()  # Blank line after left/right pair
        current_step += 1

        # Step 6: Apply AI upscaling (optional - delegated to upscaler)
        if settings.get("upscale_model", "none") != "none":
            if not self.upscaler.apply_upscaling(directories, settings, progress_tracker):
                return self._handle_step_error("Upscaling failed")
            print(
                step_complete(
                    f"Step 6: Upscaled {num_frames} frames using {settings['upscale_model']}"
                )
            )
            self._print_saved_to(directories.get("left_upscaled"), "Upscaled left frames")
            self._print_saved_to(directories.get("right_upscaled"), "Upscaled right frames")
            print()  # Blank line after left/right pair
            current_step += 1

        # Step 7: Assemble VR frames (delegated to vr_assembler)
        if not self.vr_assembler.assemble_vr_frames(
            directories, settings, progress_tracker, num_frames
        ):
            return self._handle_step_error("VR frame assembly failed")
        print(
            step_complete(
                f"Step 7: Assembled {num_frames} {settings['vr_format']} VR frames at {settings['vr_output_width']}x{settings['vr_output_height']}"
            )
        )
        self._print_saved_to(directories.get("vr_frames"), "VR frames")
        print()  # Blank line after step
        current_step += 1

        # Step 8: Create final video (delegated to video_encoder)
        vr_frames_dir = directories.get("vr_frames")
        if not vr_frames_dir:
            return self._handle_step_error("VR frames directory not found")

        success = self.video_encoder.create_video(
            vr_frames_dir,
            directories["base"],
            video_path,
            settings,
        )

        if success:
            output_filename = generate_output_filename(
                Path(video_path).name,
                settings["vr_format"],
                settings["vr_resolution"],
            )
            print(step_complete("Step 8: Created final video"))
            self._print_saved_to(directories["base"], f"Final output: {output_filename}")

        # Finalize and cleanup
        if progress_tracker and hasattr(progress_tracker, "finish"):
            progress_tracker.finish("Video processing complete")
        self._finalize_processing(success, output_path, video_path, settings, num_frames)
        return success

    def _setup_processing(
        self,
        video_path: str,
        output_dir: str,
        settings: dict[str, Any],
        video_properties: dict[str, Any],
    ) -> tuple[Path, dict[str, Path], Path | None]:
        """
        Setup processing directories and settings file.

        Args:
            video_path: Input video path
            output_dir: Output directory
            settings: Processing settings
            video_properties: Video metadata

        Returns:
            Tuple of (output_path, directories, settings_file)

        Side effects:
            - Creates directories
            - Writes settings file
            - Console output
        """
        output_path = Path(output_dir)
        directories = create_output_directories(output_path, settings["keep_intermediates"])
        batch_name = f"{Path(video_path).stem}_{int(time.time())}"
        settings_file = save_processing_settings(
            output_path, batch_name, settings, video_properties, video_path
        )

        print(f"\n{title_bar('=== Depth Surge 3D Video Processing ===')}")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        print("Using Video-Depth-Anything for temporal consistency\n")

        return output_path, directories, settings_file

    def _finalize_processing(
        self,
        success: bool,
        output_path: Path,
        video_path: str,
        settings: dict[str, Any],
        num_frames: int,
    ) -> None:
        """
        Finalize processing and update settings file.

        Args:
            success: Whether processing succeeded
            output_path: Output directory path
            video_path: Input video path
            settings: Processing settings
            num_frames: Number of frames processed

        Side effects:
            - Updates settings file with completion status
            - Console output
        """
        if success:
            # Calculate processing time
            elapsed_time = time.time() - self._start_time
            formatted_time = self._format_processing_time(elapsed_time)

            # Generate output filename
            output_filename = generate_output_filename(
                Path(video_path).name,
                settings["vr_format"],
                settings["vr_resolution"],
            )
            output_file_path = str(output_path / output_filename)

            # Display colored completion banner
            completion_banner(
                output_file=output_file_path,
                processing_time=formatted_time,
                num_frames=num_frames,
                vr_format=settings["vr_format"],
            )

            # Update settings file
            if self._settings_file:
                update_processing_status(
                    self._settings_file,
                    "completed",
                    {
                        "final_output": output_file_path,
                        "frames_processed": num_frames,
                        "processing_time_seconds": elapsed_time,
                    },
                )
        elif self._settings_file:
            update_processing_status(
                self._settings_file, "failed", {"error": "Video creation failed"}
            )

    @staticmethod
    def _get_total_steps(settings: dict[str, Any]) -> int:
        """
        PURE: Calculate total steps based on settings.

        Args:
            settings: Processing settings

        Returns:
            Total number of pipeline steps (6-8)
        """
        total = 6  # Base: Extract, Depth, Stereo, Crop, Assemble, Video
        if settings.get("apply_distortion", True):
            total += 1  # Add Step 4: Distortion
        if settings.get("upscale_model", "none") != "none":
            total += 1  # Add Step 6: Upscaling
        return total  # 6-8 steps total

    @staticmethod
    def _format_processing_time(seconds: float) -> str:
        """
        PURE: Format processing time as human-readable string.

        Args:
            seconds: Processing time in seconds

        Returns:
            Formatted time string (e.g., "1h 23m 45s", "5m 30s", "45s")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:  # Always show seconds if no other parts
            parts.append(f"{secs}s")

        return " ".join(parts)

    def _update_step_progress(
        self,
        progress_tracker,
        message: str,
        step_name: str,
        progress: int,
        total: int,
    ) -> None:
        """
        Update progress for a processing step.

        Args:
            progress_tracker: Progress tracker instance
            message: Progress message
            step_name: Name of current step
            progress: Current progress value
            total: Total progress value

        Side effects:
            - Updates progress tracker state
        """
        if progress_tracker:
            progress_tracker.update_progress(
                message,
                phase="processing",
                frame_num=progress,
                step_name=step_name,
                step_progress=progress,
                step_total=total,
            )

    def _print_step_complete(
        self, num_items: int, duration: float, item_type: str = "frames"
    ) -> None:
        """
        Print step completion message.

        Args:
            num_items: Number of items processed
            duration: Processing duration in seconds
            item_type: Type of items processed

        Side effects:
            - Console output
        """
        print(step_complete(f"Processed {num_items:04d} {item_type} in {duration:.2f}s"))

    def _print_saved_to(self, directory: Path | None, message_prefix: str = "Saved to") -> None:
        """
        Print save location message.

        Args:
            directory: Directory path or None
            message_prefix: Message prefix text

        Side effects:
            - Console output
        """
        if directory:
            print(saved_to(f"{message_prefix}: {directory}"))

    def _handle_step_error(self, error_msg: str) -> bool:
        """
        Handle step failure and update settings file.

        Args:
            error_msg: Error message

        Returns:
            False (always returns False to indicate failure)

        Side effects:
            - Console output
            - Updates settings file with error status
        """
        print(f"Error: {error_msg}")
        if self._settings_file:
            update_processing_status(self._settings_file, "failed", {"error": error_msg})
        return False
