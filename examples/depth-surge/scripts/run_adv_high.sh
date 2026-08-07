#!/bin/bash
# depth-surge 生产脚本: v2 + Video-Depth-Anything-Large @448px, 串行跑 advanced+high 填充。
# 支持: 竖屏 autorotate、平移量(视差强度)、时间切片、GPU 选择。
#
# 用法:
#   ./run_adv_high.sh <input.mp4> <outbase> [选项]
#   选项:
#     --vr-resolution <16x9-1080p|custom:WxH>   输出分辨率 (默认 16x9-1080p; 竖屏用 custom:1080x1920)
#     --fill <列表>                              填充质量 (默认 "advanced high"; 可只传一个)
#     --gpu <N>                                  用第 N 张卡 (默认 0)
#     --baseline <m>                             平移量: 基线距离米 (默认 0.065=人眼IPD)
#     --focal-length <px>                        平移量: 焦距像素 (默认 1000)
#     --depth-resolution <px>                    VDA 输入边长 (默认 448)
#     --model <path>                             VDA Large 权重路径 (默认 $DS/models/...)
#     --start <sec>  --end <sec>                 时间切片 (秒)
#
# 平移量关系: max_disp(px) ≈ focal_length × baseline × 0.19。
#   例: 0.065m×1000px≈12px; 想加深度 → --baseline 0.13 (≈25px);
#   想减 → --baseline 0.03 (≈6px)。范围: baseline 0.01-0.5, focal 100-5000。
# 竖屏素材带 rotation=90 元数据 → 脚本自动 ffmpeg autorotate 转正。
#
# 可移植性(不绑定绝对路径/素材):
#   - 仓库根由脚本位置推导(../../..)
#   - Python 解释器: 环境变量 DS_PYTHON(默认 python3; 需在装有 depth-surge 依赖的 env 里运行)
#   - 模型权重: 环境变量 MODEL 或 --model
# 深度缓存: 默认写 ~/.cache/depth-surge-3d/; 换素材/模型建议清掉, 或用 XDG_CACHE_HOME 指到运行目录。
set -e

INPUT="${1:?用法: $0 <input.mp4> <outbase> [选项]}"
OUTBASE="${2:?用法: $0 <input.mp4> <outbase> [选项]}"
shift 2

VR="16x9-1080p"
FILLS="advanced high"
GPU=0
BASELINE=0.065
FOCAL=1000
DRES=448
MODEL="${MODEL:-}"
START=""
END=""
RESUME=0

while [ $# -gt 0 ]; do
  case "$1" in
    --vr-resolution) VR="$2"; shift 2;;
    --fill)          FILLS="$2"; shift 2;;
    --gpu)           GPU="$2"; shift 2;;
    --baseline)      BASELINE="$2"; shift 2;;
    --focal-length)  FOCAL="$2"; shift 2;;
    --depth-resolution) DRES="$2"; shift 2;;
    --model)         MODEL="$2"; shift 2;;
    --start)         START="$2"; shift 2;;
    --end)           END="$2"; shift 2;;
    --resume)        RESUME=1; shift;;    # 继续处理: 复用已存在的帧/深度/VR帧
    *) echo "未知选项: $1"; exit 1;;
  esac
done

# 仓库根 = scripts/../../.. (examples/depth-surge/scripts → 仓库根)
DS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${DS_PYTHON:-python3}"
MODEL="${MODEL:-$DS/models/Video-Depth-Anything-Large/video_depth_anything_vitl.pth}"
[ -f "$MODEL" ] || { echo "ERROR: 找不到模型 $MODEL (用 --model 或 MODEL 环境变量指定)"; exit 1; }
mkdir -p "$OUTBASE"
cd "$DS"

# ---- Step 0: autorotate(竖屏 rotation=90 → 转正) + 可选切片 ----
IN="$INPUT"
ROT=$(ffprobe -v error -select_streams v:0 -show_entries stream_side_data=rotation -of default=nw=1:nk=1 "$INPUT" 2>/dev/null | tr -d '[:space:]')
if [ -n "$ROT" ] && [ "$ROT" != "0" ] && [ "$ROT" != "N/A" ]; then
  IN="$OUTBASE/_autorotated.mp4"
  if [ -f "$IN" ]; then
    echo "[0] RESUME: $IN 已存在, 跳过 autorotate"
  else
    echo "[0] 检测到 rotation=$ROT, ffmpeg autorotate 转正..."
    SS=""; [ -n "$START" ] && SS="-ss $START"
    TO=""; [ -n "$END" ] && TO="-to $((END-START))"
    # 保住源音频 (-c:a copy, 不重编码) — DS 编码时从 _autorotated.mp4 取音频。
    # 之前用 -an 把音频剥了 → DS fallback 取到无声视频 → 成品无音轨。
    ffmpeg -y -v error $SS $TO -i "$INPUT" -c:v libx264 -crf 18 -c:a copy "$IN"
  fi
  START=""; END=""   # 切片已在上面完成
fi

DISP=$(awk "BEGIN{printf \"%d\", ${FOCAL}*${BASELINE}*0.19}")   # int(focal×baseline×0.19); 用 awk 支持 float baseline
echo "== v2+Large@${DRES} baseline=${BASELINE}m focal=${FOCAL}px → max_disp≈${DISP}px, fill=[$FILLS], vr=$VR =="

for FILL in $FILLS; do
  echo "===== fill=$FILL ====="
  # 首次运行: depth_surge_3d.py 在 --output-dir 下建 <stem>_<timestamp> 子目录;
  # --resume 时: 复用该子目录 (中间产物所在), 跳过已完成的帧/深度/VR帧。
  OUTDIR="$OUTBASE/${FILL}_out"
  RESUME_FLAG=""
  if [ "$RESUME" = 1 ]; then
    JOB_DIR=$(find "$OUTDIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
    if [ -n "$JOB_DIR" ]; then
      OUTDIR="$JOB_DIR"
      RESUME_FLAG="--resume"
      echo "RESUME: 复用任务目录 $OUTDIR"
    else
      echo "RESUME: $OUTDIR 下无既有任务目录, 按全新运行"
    fi
  fi
  env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY depth_surge_3d.py "$IN" \
    --output-dir "$OUTDIR" $RESUME_FLAG \
    --format side_by_side --vr-resolution "$VR" \
    --depth-model-version v2 --model "$MODEL" \
    --depth-resolution "$DRES" --device cuda --no-distortion \
    --baseline "$BASELINE" --focal-length "$FOCAL" \
    --hole-fill-quality "$FILL" \
    $( [ -n "$START" ] && echo --start "$START" ) \
    $( [ -n "$END" ] && echo --end "$END" ) \
    > "$OUTBASE/${FILL}.log" 2>&1
done

echo "ALL DONE → $OUTBASE/*_out/*_sbs.mp4 (swap 版: ./tools/make_sbs_swapped.sh)"
