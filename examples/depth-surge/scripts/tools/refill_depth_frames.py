#!/usr/bin/env python3
"""复用已有深度图重跑立体生成(免深度重算, 只做填充)。
输入: depth-surge 一次运行目录的 00_original_frames/ + 02_depth_maps/
输出: SBS PNG 序列(每帧 = [left_filled | right_filled])
用途: 换 hole-fill-quality 或换平移量时, 跳过几分钟的 VDA 深度推理, 直接填充出片。
用法:
  python refill_depth_frames.py --src <run_dir> --out <out_dir> \
      --method advanced [--baseline 0.065] [--focal-length 1000]
"""
import sys, os, glob, argparse
import cv2
import numpy as np

# 仓库 src 由脚本位置推导(tools/ → 仓库根, 再 /src), 不绑定绝对路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../src"))
from depth_surge_3d.utils.imaging.image_processing import (
    depth_to_disparity,
    create_shifted_image,
    hole_fill_image,
)

def imread_robust(p):
    buf = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="depth-surge 运行目录(含 00_original_frames/ 和 02_depth_maps/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True, choices=["advanced", "high"])
    ap.add_argument("--baseline", type=float, default=0.065, help="平移量: 基线米(默认0.065)")
    ap.add_argument("--focal-length", type=float, default=1000, help="平移量: 焦距像素(默认1000)")
    args = ap.parse_args()

    orig_dir = os.path.join(args.src, "00_original_frames")
    depth_dir = os.path.join(args.src, "02_depth_maps")
    frames = sorted(glob.glob(os.path.join(orig_dir, "frame_*.png")))
    if not frames:
        sys.exit(f"ERROR: 无帧文件 in {orig_dir}")
    os.makedirs(args.out, exist_ok=True)
    print(f"复用 {len(frames)} 帧 + 深度图, fill={args.method}, "
          f"baseline={args.baseline}m focal={args.focal_length}px", flush=True)

    for i, fp in enumerate(frames):
        base = os.path.basename(fp)
        depth_path = os.path.join(depth_dir, base)
        frame = imread_robust(fp)
        depth = imread_robust(depth_path)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        disp = depth_to_disparity(depth, args.baseline, args.focal_length)
        left = create_shifted_image(frame, disp, "left")
        right = create_shifted_image(frame, disp, "right")
        left = hole_fill_image(left, method=args.method)
        right = hole_fill_image(right, method=args.method)
        sbs = np.hstack([left, right])
        cv2.imwrite(os.path.join(args.out, f"{i:06d}.png"), sbs)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frames)}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
