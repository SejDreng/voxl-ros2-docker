#!/bin/bash
set -e

echo "[entrypoint] Fixing libgbm symlink..."
if [ -e /opt/voxl-lib64/libgbm.so ]; then
	rm -rf /opt/voxl-lib64/libgbm.so
fi
ln -s libgbm.so.gbm /opt/voxl-lib64/libgbm.so

echo "[entrypoint] Starting iox-roudi in background..."
source /ros2_ws/install/setup.bash
iox-roudi &
IOX_PID=$!
echo "[entrypoint] iox-roudi started (PID $IOX_PID)"

exec "$@"
