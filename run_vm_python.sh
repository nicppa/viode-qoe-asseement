#!/usr/bin/env bash

set -euo pipefail

print_usage() {
  cat <<'USAGE'
用法:
  run_vm_python.sh \
    -h <host> \
    -u <user> \
    -p <password> \
    -f <local_py_file> \
    [-P <port=22>] \
    [-r <remote_dir=/tmp/remote_py>] \
    [-b <python_bin=python3>] \
    [-- <args...>]

示例:
  run_vm_python.sh -h 127.0.0.1 -P 2222 -u vbox -p 'yourPass' -f /path/to/app.py -- --foo bar 123
USAGE
}

vm_host="172"
ssh_port="22"
vm_user=""
vm_password=""
local_py_file=""
remote_dir="/tmp/remote_py"
python_bin="python3"
py_args=()

if [[ $# -eq 0 ]]; then
  print_usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--host)
      [[ $# -ge 2 ]] || { echo "缺少 -h/--host 参数值" >&2; exit 2; }
      vm_host="$2"; shift 2 ;;
    -P|--port)
      [[ $# -ge 2 ]] || { echo "缺少 -P/--port 参数值" >&2; exit 2; }
      ssh_port="$2"; shift 2 ;;
    -u|--user)
      [[ $# -ge 2 ]] || { echo "缺少 -u/--user 参数值" >&2; exit 2; }
      vm_user="$2"; shift 2 ;;
    -p|--password)
      [[ $# -ge 2 ]] || { echo "缺少 -p/--password 参数值" >&2; exit 2; }
      vm_password="$2"; shift 2 ;;
    -f|--file)
      [[ $# -ge 2 ]] || { echo "缺少 -f/--file 参数值" >&2; exit 2; }
      local_py_file="$2"; shift 2 ;;
    -r|--remote-dir)
      [[ $# -ge 2 ]] || { echo "缺少 -r/--remote-dir 参数值" >&2; exit 2; }
      remote_dir="$2"; shift 2 ;;
    -b|--python-bin)
      [[ $# -ge 2 ]] || { echo "缺少 -b/--python-bin 参数值" >&2; exit 2; }
      python_bin="$2"; shift 2 ;;
    --)
      shift
      # 余下均为传给 Python 的参数
      while [[ $# -gt 0 ]]; do py_args+=("$1"); shift; done
      ;;
    -?|--help)
      print_usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      print_usage
      exit 2 ;;
  esac
done

# 校验依赖与必需参数
if ! command -v sshpass >/dev/null 2>&1; then
  echo "未找到 sshpass，请先安装：macOS 可用 'brew install hudochenkov/sshpass/sshpass' 或使用其他安装方式" >&2
  exit 3
fi

[[ -n "$vm_host" ]] || { echo "必须提供 -h/--host" >&2; exit 2; }
[[ -n "$vm_user" ]] || { echo "必须提供 -u/--user" >&2; exit 2; }
[[ -n "$vm_password" ]] || { echo "必须提供 -p/--password" >&2; exit 2; }
[[ -n "$local_py_file" ]] || { echo "必须提供 -f/--file" >&2; exit 2; }

if [[ ! -f "$local_py_file" ]]; then
  echo "本地文件不存在: $local_py_file" >&2
  exit 4
fi

remote_basename="$(basename -- "$local_py_file")"
remote_path="$remote_dir/$remote_basename"

ssh_opts=(
  -o StrictHostKeyChecking=no
  -p "$ssh_port"
)

# 创建远程目录
sshpass -p "$vm_password" \
  ssh "${ssh_opts[@]}" "$vm_user@$vm_host" \
  "mkdir -p \"$remote_dir\""

# 复制文件至远程
sshpass -p "$vm_password" \
  scp -q -P "$ssh_port" -o StrictHostKeyChecking=no \
  "$local_py_file" "$vm_user@$vm_host:$remote_path"

# 拼接远程执行参数，做安全转义（单引号转义为 '\''）
quoted_args=""
if [[ ${#py_args[@]} -gt 0 ]]; then
  for arg in "${py_args[@]}"; do
    escaped=${arg//\'/\'\\\'\'}
    quoted_args+=" '${escaped}'"
  done
fi

# 执行远程 Python
remote_cmd="$python_bin '$remote_path'$quoted_args"

sshpass -p "$vm_password" \
  ssh "${ssh_opts[@]}" "$vm_user@$vm_host" \
  "$remote_cmd"



