#!/bin/bash

# 检查输入参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_json>"
    exit 1
fi

input_folder=$1
output_json=$2

# 检查输入文件夹是否存在
if [ ! -d "$input_folder" ]; then
    echo "Input folder not found: $input_folder"
    exit 1
fi

# 查找文件夹中的所有图片文件，并将文件名存储到数组中
files=($(find "$input_folder" -type f -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.gif" | sort))

# 构建JSON对象
json="["
for file in "${files[@]}"; do
    filename=$(basename "${file%????}")
    json+="\"$filename\", "
done
# 移除最后一个逗号和空格
json="${json%??}"
json+="]"

# 将JSON对象写入文件
echo "$json" > "$output_json"

echo "Images list saved to: $output_json"
