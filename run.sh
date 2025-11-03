#!/usr/bin/env bash
# No `set -u` before sourcing venv to avoid LD_LIBRARY_PATH errors
set -eE -o pipefail

cd "$HOME/PArobot"

# Short device wait loop (max ~8s) for camera + LiDAR
for i in {1..8}; do
  READY=1
  [[ -c /dev/video0 ]]  || READY=0   # camera
  [[ -e /dev/ttyUSB0 ]] || READY=0   # YDLIDAR on USB
  if [[ $READY -eq 1 ]]; then break; fi
  sleep 1
done

# Activate venv and run
source "$HOME/PArobot/capstone/bin/activate"

# Run your script (edit args if paths differ)
exec python3 main3.py --modeldir . 
