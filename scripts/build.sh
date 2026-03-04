#!/usr/bin/env bash
# =============================================================================
# build.sh — Multi-arch, multi-stage build, cross-compile, and deploy helper
#
# Images:
#   dev-amd64       Full dev image (workstation, x86_64)
#   dev-arm64       Full dev image (cross-build via QEMU, arm64)
#   runtime-arm64   Slim runtime image (drone, arm64)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="${PROJECT_DIR}/docker"

IMAGE_NAME="voxl-drone"

# Drone SSH connection (override with env vars or .env file)
VOXL_USER="${VOXL_USER:-ubuntu}"
VOXL_HOST="${VOXL_HOST:-drone.local}"
VOXL_DIR="${VOXL_DIR:-/voxl_docker}"

# Load .env if present
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

help() {
    cat <<EOF
Help: $(basename "$0") <command>

---- HELP ----
  help                 Display all commands

---- ONE-TIME SETUP ----
  setup-qemu           Install QEMU user-static for arm64 emulation (run once)

---- BUILD IMAGES ----
  build-deps           Build only the dependency base stage (useful to verify deps)
  build-dev            Build the full dev image (native x86_64)
  build-cross          Build the full dev image for arm64 via QEMU
  build-runtime        Build the slim runtime image for arm64
  clean-build          Erase build images, containers, builders, and artifacts to allow for a complete rebuild

---- DEVELOPMENT (workstation) ----
  dev                  Open a shell in the native x86 dev container
  cross                Open a shell in the arm64 QEMU dev container
  build-ws             Run colcon build in the native dev container
  build-ws-cross       Run colcon build in the arm64 container (produces arm64 binaries)

---- DEPLOY TO DRONE ----
  export-runtime       Save the slim runtime image to a .tar.gz file
  extract-install      Copy cross-built arm64 install/ out of the Docker volume
  deploy               Rsync source + install + compose to drone
  deploy-image         Transfer the runtime image .tar.gz to drone and load it

---- DRONE OPERATIONS (via SSH) ----
  voxl-start          Start the voxl-drone container
  voxl-shell          Attach to the running voxl-drone container
  voxl-logs           Show voxl-drone container logs
  voxl-stop           Stop the voxl-drone container
EOF
}

# ============================= ONE-TIME SETUP ================================

cmd_setup_qemu() {
    echo "==> Installing QEMU user-static for multi-arch support..."
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    echo "==> Creating buildx builder..."
    docker buildx create --name multiarch --driver docker-container --use 2>/dev/null || \
        docker buildx use multiarch
    docker buildx inspect --bootstrap
    echo ""
    echo "==> Done. You can now build arm64 images on this x86 machine."
}

# ============================== BUILD IMAGES =================================

cmd_build_deps() {
    echo "==> Building dependency base image (voxl-deps)..."
    docker buildx build \
        --platform linux/amd64 \
        --target voxl-deps \
        -f "${DOCKER_DIR}/Dockerfile" \
        -t "${IMAGE_NAME}:deps-amd64" \
        --load \
        "${DOCKER_DIR}"
    echo "==> Built: ${IMAGE_NAME}:deps-amd64"
}

cmd_build_dev() {
    echo "==> Building dev image for x86_64 (voxl-dev)..."
    docker buildx build \
        --platform linux/amd64 \
        --target voxl-dev \
        -f "${DOCKER_DIR}/Dockerfile" \
        -t "${IMAGE_NAME}:dev-amd64" \
        --load \
        "${DOCKER_DIR}"
    echo "==> Built: ${IMAGE_NAME}:dev-amd64"
    echo "    Image size:"
    docker images "${IMAGE_NAME}:dev-amd64" --format "    {{.Size}}"
}

cmd_build_cross() {
    echo "==> Building dev image for arm64 via QEMU (voxl-dev)..."
    docker buildx build \
        --platform linux/arm64 \
        --target voxl-dev \
        -f "${DOCKER_DIR}/Dockerfile" \
        -t "${IMAGE_NAME}:dev-arm64" \
        --load \
        "${DOCKER_DIR}"
    echo "==> Built: ${IMAGE_NAME}:dev-arm64"
    echo "    Image size:"
    docker images "${IMAGE_NAME}:dev-arm64" --format "    {{.Size}}"
}

cmd_build_runtime() {
    echo "==> Building slim runtime image for arm64 (STAGE 2: drone-runtime)..."
    docker buildx build \
        --platform linux/arm64 \
        --target voxl-runtime \
        -f "${DOCKER_DIR}/Dockerfile" \
        -t "${IMAGE_NAME}:runtime-arm64" \
        --load \
        "${DOCKER_DIR}"
    echo "==> Built: ${IMAGE_NAME}:runtime-arm64"
    echo "    Image size:"
    docker images "${IMAGE_NAME}:runtime-arm64" --format "    {{.Size}}"
}

cmd_clean_build() {
    echo "==> Stopping all project containers..."
    docker compose -f "${DOCKER_DIR}/docker-compose_workstation.yml" down --remove-orphans 2>/dev/null || true

    echo "==> Removing project images..."
    docker images --filter "reference=${IMAGE_NAME}:*" -q | xargs -r docker rmi -f

    echo "==> Removing named volumes..."
    for vol in dev-build dev-install dev-log cross-build cross-install cross-log; do
        docker volume rm "$vol" 2>/dev/null && echo "    Removed volume: $vol" || true
    done

    echo "==> Stopping and removing buildx builders..."
    docker buildx stop multiarch 2>/dev/null || true
    docker buildx rm multiarch 2>/dev/null || true

    echo "==> Pruning build cache..."
    docker builder prune -af

    echo "==> Pruning dangling images and stopped containers..."
    docker system prune -af

    echo "==> Removing exported image tarball (if any)..."
    rm -f "${PROJECT_DIR}/voxl-runtime-arm64.tar.gz"

    echo ""
    echo "==> Clean complete. Run 'setup-qemu' again if you need cross-builds."
}

# ============================= DEVELOPMENT ===================================

cmd_dev() {
    echo "==> Starting native x86_64 dev container..."
    docker compose -f "${DOCKER_DIR}/docker-compose.workstation.yml" run --rm dev
}

cmd_cross() {
    echo "==> Starting arm64 cross-build container (QEMU)..."
    docker compose -f "${DOCKER_DIR}/docker-compose.workstation.yml" run --rm cross-arm64
}

cmd_build_ws() {
    echo "==> Running colcon build in native dev container..."
    docker compose -f "${DOCKER_DIR}/docker-compose.workstation.yml" run --rm dev \
        bash -c "source /opt/ros/humble/setup.bash && cd /ros2_ws && colcon build --symlink-install"
}

cmd_build_ws_cross() {
    echo "==> Running colcon build in arm64 container (QEMU-emulated)..."
    docker compose -f "${DOCKER_DIR}/docker-compose.workstation.yml" run --rm cross-arm64 \
        bash -c "source /opt/ros/humble/setup.bash && colcon build"
    echo ""
    echo "==> ARM64 binaries built. Run 'extract-install' to copy them out."
}

# =============================== DEPLOY ======================================

cmd_export_runtime() {
    local outfile="${PROJECT_DIR}/voxl-runtime-arm64.tar.gz"
    echo "==> Exporting runtime image to ${outfile}..."
    docker save "${IMAGE_NAME}:runtime-arm64" | gzip > "${outfile}"
    local size
    size=$(du -h "${outfile}" | cut -f1)
    echo "==> Saved: ${outfile} (${size})"
}

cmd_extract_install() {
    local dest="${PROJECT_DIR}/deploy/install"
    rm -rf "${dest}"
    mkdir -p "${dest}"

    echo "==> Extracting arm64 install/ from cross-build volume..."

    # Use existing image to access the name volume
    docker run --rm \
        -v cross-install:/src:ro \
        -v "${dest}:/dest" \
        "${IMAGE_NAME}:dev-amd64" \
        bash -c "cp -a /src/. /dest/"

    echo "==> Extracted to ${dest}/"
    echo "    Contents:"
    ls "${dest}/" 2>/dev/null || echo "    (empty — have you run build-ws-cross?)"
}

cmd_deploy() {
    local deploy_dir="${PROJECT_DIR}/deploy"

    echo "==> Preparing deploy directory..."
    mkdir -p "${deploy_dir}"

    # Stop the running container before syncing new files
    echo "==> Stopping drone container (if running)..."
    ssh "${VOXL_USER}@${VOXL_HOST}" \
        "cd ${VOXL_DIR} 2>/dev/null && docker compose down 2>/dev/null || true"

    # Copy compose file for voxl drone
    cp "${DOCKER_DIR}/docker-compose.voxl.yml" "${deploy_dir}/docker-compose.yml"

    # Copy source files
    rsync -a --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        "${PROJECT_DIR}/ros2_ws/src/" "${deploy_dir}/src/"

    # Check install/ exists
    if [ ! -d "${deploy_dir}/install" ] || [ -z "$(ls -A "${deploy_dir}/install" 2>/dev/null)" ]; then
        echo "    WARNING: deploy/install/ is empty. Run 'extract-install' first."
        echo "             (Or the drone can build from source if you prefer.)"
    fi

    # Copy install files
    echo "==> Syncing to ${VOXL_USER}@${VOXL_HOST}:${VOXL_DIR}..."
    ssh "${VOXL_USER}@${VOXL_HOST}" "mkdir -p ${VOXL_DIR}" # Ensure VOXL_DIR exists on drone
    rsync -avz --progress --delete \
        "${deploy_dir}/" \
        "${VOXL_USER}@${VOXL_HOST}:${VOXL_DIR}/"

    echo ""
    echo "==> Deploy complete. Drone directory layout:"
    echo "    ${VOXL_DIR}/"
    echo "    ├── docker-compose.yml"
    echo "    ├── src/               (ROS2 packages)"
    echo "    └── install/           (pre-built arm64 binaries)"
}

cmd_deploy_image() {
    local image_file="${PROJECT_DIR}/voxl-runtime-arm64.tar.gz"

    if [ ! -f "${image_file}" ]; then
        echo "==> Runtime image not exported yet. Building and exporting..."
        cmd_build_runtime
        cmd_export_runtime
    fi

    echo "==> Transferring runtime image to drone..."
    rsync -avz --progress "${image_file}" "${VOXL_USER}@${VOXL_HOST}:/tmp/"

    echo "==> Loading image on drone..."
    ssh "${VOXL_USER}@${VOXL_HOST}" "docker load < /tmp/voxl-runtime-arm64.tar.gz && rm /tmp/voxl-runtime-arm64.tar.gz"

    echo "==> Done. Image loaded on drone."
}

# =========================== DRONE OPERATIONS ================================

cmd_voxl_start() {
    echo "==> Starting drone container on ${VOXL_HOST}..."
    ssh -t "${VOXL_USER}@${VOXL_HOST}" \
        "cd ${VOXL_DIR} && docker compose up -d"
}

cmd_voxl_shell() {
    echo "==> Connecting to drone container..."
    ssh -t "${VOXL_USER}@${VOXL_HOST}" \
        "docker exec -it voxl-runtime bash"
}

cmd_voxl_logs() {
    ssh "${VOXL_USER}@${VOXL_HOST}" \
        "cd ${VOXL_DIR} && docker compose logs -f --tail=100"
}

cmd_voxl_stop() {
    echo "==> Stopping voxl-drone container..."
    ssh -t "${VOXL_USER}@${VOXL_HOST}" \
        "cd ${VOXL_DIR} && docker compose down"
}

# =============================================================================
case "${1:-}" in
    setup-qemu)       cmd_setup_qemu ;;
    build-deps)       cmd_build_deps ;;
    build-dev)        cmd_build_dev ;;
    build-cross)      cmd_build_cross ;;
    build-runtime)    cmd_build_runtime ;;
    clean-build)      cmd_clean_build ;; 
    dev)              cmd_dev ;;
    cross)            cmd_cross ;;
    build-ws)         cmd_build_ws ;;
    build-ws-cross)   cmd_build_ws_cross ;;
    export-runtime)   cmd_export_runtime ;;
    extract-install)  cmd_extract_install ;;
    deploy)           cmd_deploy ;;
    deploy-image)     cmd_deploy_image ;;
    voxl-start)       cmd_voxl_start ;;
    voxl-shell)       cmd_voxl_shell ;;
    voxl-logs)        cmd_voxl_logs ;;
    voxl-stop)        cmd_voxl_stop ;;
    help)             help ;;
    *)                help ;;
esac
