#!/bin/bash
# run_openscad.sh

# 设置依赖库路径
export LD_LIBRARY_PATH="/home/yc27979/xa/openscad_verl/usr/lib/x86_64-linux-gnu"

# 执行 OpenSCAD 命令 分别传入output_stl 和 input_scad 参数
$1 -o "$2" --backend Manifold "$3"