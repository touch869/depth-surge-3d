#!/bin/bash
# 生成 SWAPPED 版(左右眼互换, 用户交叉眼观看时近物出屏方向正确)。
# 两方案(depth-surge/StereoCrafter)输出都是"左右排列"的 Full-SBS,
# 横屏(如 3840x1080→每半1920x1080)竖屏(如 2160x1920→每半1080x1920)通用。
# 用法: ./make_sbs_swapped.sh <input_sbs.mp4> [output.mp4]
#   缺省输出: <输入去扩展名>_SWAPPED.mp4, 保留音轨。
set -e
IN="${1:?用法: $0 <input_sbs.mp4> [output.mp4]}"
OUT="${2:-${IN%.*}_SWAPPED.mp4}"

W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$IN")
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$IN")
HW=$((W/2))

FC="[0:v]crop=${HW}:${H}:0:0[l];[0:v]crop=${HW}:${H}:${HW}:0[r];[r][l]hstack=2[v]"
ffmpeg -y -v error -i "$IN" -filter_complex "$FC" \
  -map "[v]" -map 0:a? -c:v libx264 -crf 16 "$OUT"
echo "SWAPPED → $OUT ($W x $H)"
