# depth-surge-3d 使用方案（定档版）

基于 [depth-surge-3d](https://github.com/piiswrong/depth-surge-3d)（深度估计 + DIBR 立体生成），在 RTX 3090 24GB 上实测敲定的 2D→3D 方案。**源码修复已直接提交在本 fork**(见下文「源码改动」), 无需再打 patch。

## 定档规格

| 项 | 定档值 | 说明 |
|---|---|---|
| 深度模型 | **V2 + Video-Depth-Anything Large @448px 输入** | 518px 在 24GB OOM(>20GB); 448px 峰值 13.97GB 可行 |
| 输出分辨率 | 1080p 横屏 `custom:1080x1920`(竖屏) | 需本 fork 的 `--vr-resolution` 放开(custom:WxH) |
| 填充 | **advanced + high 两版** | adv 有深度层次; high 偏均匀 |
| 方向 | **swap + 非swap 两版** | 交叉眼观看用 swap=近物出屏正确方向 |
| 产出 | **每素材 4 文件** | `{adv,high} × {sbs, sbs_SWAPPED}` |
| 平移量 | `--baseline`(m, 默认0.065=IPD) + `--focal-length`(px, 默认1000) | `max_disp≈focal×baseline×0.19≈12px`, 线性 |

## 关键结论（实测）

1. **v2(VDA 时序) > v3(DA3 逐帧)**: v3 是全局均匀平移(块间视差 std≈1.0), 无真实深度结构 → "纸片"无立体感。v2 逐像素深度变化(std 1.6-2.1)才有真实立体。**立体感指标 = 块间视差 std, 不是 MAD 全局位移**。
2. **深度缓存撞 key 坑**: v2/v3 时 cache key settings 全 null → 撞 key 误用旧深度。**换模型/素材必须清深度缓存**(`~/.cache/depth-surge-3d/`, 或运行前 `export XDG_CACHE_HOME=<运行目录>/cache` 隔离)。
3. **并行写同目录互覆**: adv/high 并行跑同 input 同 output-dir → 中间帧互相覆盖结果作废。**必须独立输出目录 + 串行**(run_adv_high.sh 已串行)。
4. **竖屏视频带 rotation=90**: 用 **ffmpeg autorotate**(`scale` 默认自动应用 rotation), 不要手动 transpose(会在错误基准上转90° → 画面倾倒)。run_adv_high.sh 已自动处理。
5. **复用深度图提速**: 深度图在 run 目录 `02_depth_maps/`, 换填充/平移量时可直接复用跳过深度重算(`scripts/tools/refill_depth_frames.py`)。

## 目录结构

```
examples/depth-surge/
├── README.md                    本文件
└── scripts/
    ├── run_adv_high.sh           生产脚本(推荐): v2+Large@448 串行跑 adv+high,
    │                             支持 autorotate / 平移量(baseline,focal) / 切片 / GPU
    └── tools/
        ├── make_sbs_swapped.sh   生成 SWAPPED 版(左右眼互换, 横竖屏自适应)
        └── refill_depth_frames.py 复用已有深度图重跑填充(免深度重算, 可换填充/平移量)
```

## 源码改动（已 commit, 无需 patch）

本 fork 在 `f5360f2` 提交了 9 个文件的修复, 直接用即可:

| 文件 | 改动 |
|---|---|
| `depth_surge_3d.py` | `--vr-resolution` choices→type=str, 支持 `custom:WxH`(竖屏); 新增 `--depth-resolution` |
| `src/.../utils/imaging/image_processing.py` | **`depth_to_disparity` 修复**: 归一化深度被当米制→视差2800px爆表→黑影+裁切; 改有界像素映射 `disp=(1-depth)*max_disp, max_disp≈focal*baseline*0.19≈12px` |
| `src/.../processing/orchestration/pipeline_orchestrator.py` | 新增 NullProgressTracker(CLI 模式 progress_tracker=None 崩) |
| `src/.../processing/frames/stereo_generator.py` | progress_tracker 守卫 + 串行立体生成(62-worker fork 崩) |
| `src/.../processing/frames/depth_processor.py` | v2 的 input_size 上限 518(防 OOM) |
| `src/.../utils/system/vram_manager.py` | v2 每帧显存开销 + max_chunk 调整 |
| `src/.../core/constants.py` | super_sample "auto"→"none"(4K upscale 形状错) + depth_resolution 配置 |
| `src/.../processing/video/video_encoder.py` | 强制 libx264(NVENC preset p7 不兼容) |
| `src/.../rendering/stereo_projector.py` | process_video 加 depth_resolution 参数 |

## 运行（推荐: 生产脚本）

```bash
# 前置: 在装有 depth-surge 依赖的 env 里运行(本 fork 已含源码修复)
# 可选: export DS_PYTHON=/path/to/env/bin/python ; export MODEL=/path/to/video_depth_anything_vitl.pth

# 横屏 1080p, v2+Large@448, adv+high 两填充
./scripts/run_adv_high.sh input.mp4 out/

# 竖屏(自动 autorotate 转正 1080x1920)
./scripts/run_adv_high.sh input.mp4 out/ --vr-resolution custom:1080x1920

# 加平移量(视差强度): baseline 0.13 ≈ max_disp 25px; 只跑 advanced; 用 GPU1
./scripts/run_adv_high.sh input.mp4 out/ --baseline 0.13 --fill advanced --gpu 1

# 切片: 只处理 30~45 秒
./scripts/run_adv_high.sh input.mp4 out/ --start 30 --end 45

# 平移量关系: max_disp(px) ≈ focal_length × baseline × 0.19
#   默认 0.065m×1000px≈12px; baseline 可调 0.01-0.5, focal 100-5000
```

## swap 生成（工具脚本）

```bash
# 横竖屏自适应, 自动探测半宽/半高
./scripts/tools/make_sbs_swapped.sh out_adv_sbs.mp4          # → out_adv_sbs_SWAPPED.mp4
./scripts/tools/make_sbs_swapped.sh out.mp4 out_SWAPPED.mp4
```

## 复用深度图提速

```bash
# 换填充质量或平移量时跳过深度重算(直接读已有 00_original_frames/ + 02_depth_maps/)
python scripts/tools/refill_depth_frames.py --src <上次run目录> --out refill_out \
  --method advanced --baseline 0.13
# 注: 平移量/填充变化不影响深度图, 可复用; 换素材则不能复用(须清深度缓存)
```

## 直接调用 CLI（原始命令）

```bash
# 例: v2+Large@448 advanced 填充(需本 fork 源码 + VDA Large 权重)
python depth_surge_3d.py input.mp4 \
  --output-dir out/adv --format side_by_side \
  --vr-resolution 16x9-1080p \
  --depth-model-version v2 \
  --model models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth \
  --depth-resolution 448 --device cuda --no-distortion \
  --baseline 0.065 --focal-length 1000 \
  --hole-fill-quality advanced

# 竖屏: --vr-resolution custom:1080x1920
# high 填充: --hole-fill-quality high
# swap 版: ./scripts/tools/make_sbs_swapped.sh out_sbs.mp4
```
