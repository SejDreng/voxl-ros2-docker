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
VOXL_COMPOSE_CMD="${VOXL_COMPOSE_CMD:-auto}"

REMOTE_COMPOSE_BIN=""

# Load .env if present
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

resolve_remote_compose_bin() {
    if [ -n "${REMOTE_COMPOSE_BIN}" ]; then
        echo "${REMOTE_COMPOSE_BIN}"
        return
    fi

    if [ "${VOXL_COMPOSE_CMD}" != "auto" ]; then
        REMOTE_COMPOSE_BIN="${VOXL_COMPOSE_CMD}"
        echo "${REMOTE_COMPOSE_BIN}"
        return
    fi

    if ssh "${VOXL_USER}@${VOXL_HOST}" "docker compose version >/dev/null 2>&1"; then
        REMOTE_COMPOSE_BIN="docker compose"
    elif ssh "${VOXL_USER}@${VOXL_HOST}" "command -v docker-compose >/dev/null 2>&1"; then
        REMOTE_COMPOSE_BIN="docker-compose"
    else
        echo "==> ERROR: No Docker Compose command found on drone (tried 'docker compose' and 'docker-compose')." >&2
        return 1
    fi

    echo "==> Using remote compose command: ${REMOTE_COMPOSE_BIN}" >&2
    echo "${REMOTE_COMPOSE_BIN}"
}

run_remote_compose() {
    local compose_args="$1"
    local compose_bin
    compose_bin="$(resolve_remote_compose_bin)" || return 1
    ssh -t "${VOXL_USER}@${VOXL_HOST}" \
        "cd ${VOXL_DIR} && ${compose_bin} -f docker-compose.yml ${compose_args}"
}

help() {
    cat <<EOF
Help: $(basename "$0") <command>

---- HELP ----
  help                 Display all commands

---- ONE-TIME SETUP ----
  setup-qemu           Install QEMU user-static for arm64 emulation (run once)
    setup-voxl-services  Install and enable FastDDS + voxl_mpa_to_ros2 systemd services on VOXL

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
  deploy               Extract cross-built ARM64 install and rsync install + configs to drone
  export-runtime       Save the slim runtime image to a .tar.gz file
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

cmd_setup_voxl_services() {
    local services_dir="${PROJECT_DIR}/deploy/systemd"
    local env_file="${services_dir}/voxl-ros2-network.env"
    local discovery_service="${services_dir}/fastdds-discovery.service"
    local bridge_service="${services_dir}/voxl-mpa-to-ros2.service"

    if [ ! -f "${env_file}" ] || [ ! -f "${discovery_service}" ] || [ ! -f "${bridge_service}" ]; then
        echo "==> ERROR: Missing one or more required files in ${services_dir}" >&2
        return 1
    fi

    echo "==> Copying systemd files to ${VOXL_USER}@${VOXL_HOST}..."
    scp "${env_file}" "${VOXL_USER}@${VOXL_HOST}:/tmp/voxl-ros2-network.env"
    scp "${discovery_service}" "${VOXL_USER}@${VOXL_HOST}:/tmp/fastdds-discovery.service"
    scp "${bridge_service}" "${VOXL_USER}@${VOXL_HOST}:/tmp/voxl-mpa-to-ros2.service"

    echo "==> Installing and enabling services on drone..."
    ssh "${VOXL_USER}@${VOXL_HOST}" '
        set -e
        if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=""; fi

        $SUDO install -m 0644 /tmp/voxl-ros2-network.env /etc/default/voxl-ros2-network
        $SUDO install -m 0644 /tmp/fastdds-discovery.service /etc/systemd/system/fastdds-discovery.service
        $SUDO install -m 0644 /tmp/voxl-mpa-to-ros2.service /etc/systemd/system/voxl-mpa-to-ros2.service

        $SUDO systemctl daemon-reload
        $SUDO systemctl enable --now fastdds-discovery.service
        $SUDO systemctl enable --now voxl-mpa-to-ros2.service

        $SUDO systemctl --no-pager --full status fastdds-discovery.service | cat
        $SUDO systemctl --no-pager --full status voxl-mpa-to-ros2.service | cat
    '

    echo "==> VOXL services installed and enabled."
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
    docker compose -f "${DOCKER_DIR}/docker-compose.workstation.yml" down --remove-orphans 2>/dev/null || true

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
    echo "==> Granting container access to X server..."
    xhost +local:docker
    echo "==> Starting native x86_64 dev container..."
    docker compose -f "${DOCKER_DIR}/docker-compose.workstation.yml" run --rm dev
}

cmd_cross() {
    echo "==> Starting arm64 cross-build container (QEMU)..."
    docker compose -f "${DOCKER_DIR}/docker-compose.workstation.yml" run --rm cross-arm64
}

cmd_build_ws() {
    local colcon_args="--symlink-install"
    if [ $# -gt 0 ]; then
        colcon_args+=" --packages-select $*"
    fi

    echo "==> Running colcon build in native dev container..."
    docker compose -f "${DOCKER_DIR}/docker-compose.workstation.yml" run --rm dev \
        bash -c "source /opt/ros/humble/setup.bash && cd /ros2_ws && colcon build ${colcon_args}"
}

cmd_build_ws_cross() {
    # local colcon_args="--symlink-install"
    local colcon_args=""
    if [ $# -gt 0 ]; then
        colcon_args+=" --packages-select $*"
    fi

    echo "==> Running colcon build in arm64 container (QEMU-emulated)..."
    docker compose -f "${DOCKER_DIR}/docker-compose.workstation.yml" run --rm cross-arm64 \
        bash -c "source /opt/ros/humble/setup.bash && colcon build ${colcon_args}"
    echo ""
    echo "==> ARM64 binaries built. Run 'deploy' to deploy to drone."
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

cmd_deploy() {
    local deploy_dir="${PROJECT_DIR}/deploy"
    local ros2_install="${deploy_dir}/ros2_ws/install"

    sudo chown -R "$(id -u):$(id -g)" "${deploy_dir}" 2>/dev/null || true

    # clean deploy dir prior to extract
    cd "${deploy_dir}" && rm -fr ./* # OBS TODO: SKIP DELETE OF DATA FOLDER FOR OUTPUTS

    echo "==> Preparing deploy directory..."
    mkdir -p "${ros2_install}"


    # Open QEMU ARM64 container and copy install and source files and resolve symlinks
    echo "==> Extracting ARM64 install from cross-build volume..."
    docker run --rm \
        --platform linux/arm64 \
        -v cross-install:/cross_build/install_src:ro \
        -v cross-build:/ros2_ws/build:ro \
        -v "${PROJECT_DIR}/ros2_ws/src:/ros2_ws/src:ro" \
        -v "${ros2_install}:/ros2_install_dest" \
        "${IMAGE_NAME}:dev-arm64" \
        bash -c "cp -aL /cross_build/install_src/. /ros2_install_dest/"
    if [ -z "$(ls -A "${ros2_install}" 2>/dev/null)" ]; then
        echo "    (empty — have you run build-ws-cross?)"
    else
        echo "==> Extracted!"
    fi

    # Stop the running container before syncing new files
    echo "==> Stopping drone container (if running)..."
    local compose_bin
    compose_bin="$(resolve_remote_compose_bin)" || true
    if [ -n "${compose_bin}" ]; then
        ssh "${VOXL_USER}@${VOXL_HOST}" \
            "cd ${VOXL_DIR} 2>/dev/null && ${compose_bin} -f docker-compose.yml down 2>/dev/null || true"
    fi

    echo "==> Beginning transfer to VOXL2..."

    # Copy compose file and cyclonedds config
    cp "${DOCKER_DIR}/docker-compose.voxl.yml" "${deploy_dir}/docker-compose.yml"
    cp "${DOCKER_DIR}/cyclonedds.xml" "${deploy_dir}/cyclonedds.xml"
    cp "${DOCKER_DIR}/entrypoint.sh" "${deploy_dir}/entrypoint.sh"

    # Copy source files
    # rsync -a --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    #     "${PROJECT_DIR}/ros2_ws/src/" "${deploy_dir}/ros2_ws/src/"

    # Copy install files
    echo "==> Syncing to ${VOXL_USER}@${VOXL_HOST}:${VOXL_DIR}..."
    ssh "${VOXL_USER}@${VOXL_HOST}" "mkdir -p \"${VOXL_DIR}\"" # Ensure VOXL_DIR exists on drone
    rsync -avz --progress --delete \
        "${deploy_dir}/" \
        "${VOXL_USER}@${VOXL_HOST}:${VOXL_DIR}/"

    echo ""
    echo "==> Deploy complete. Drone directory layout:"
    echo "    ${VOXL_DIR}/"
    echo "    ├── docker-compose.yml"
    echo "    ├── cyclonedds.xml"
    echo "    └── ros2_ws"
    echo "        └── install/          (pre-built arm64 binaries)"
}

cmd_deploy_image() {
    local image_file="${PROJECT_DIR}/voxl-runtime-arm64.tar.gz"

    if ! docker image inspect "${IMAGE_NAME}:runtime-arm64" >/dev/null 2>&1; then
        echo "==> Runtime image not built yet. Building now..."
        cmd_build_runtime
    fi

    echo "==> Exporting latest runtime image tarball..."
    cmd_export_runtime

    echo "==> Transferring runtime image to drone..."
    rsync -avz --progress "${image_file}" "${VOXL_USER}@${VOXL_HOST}:/tmp/"

    echo "==> Loading image on drone... (This may take a while)"
    ssh "${VOXL_USER}@${VOXL_HOST}" "docker load < /tmp/voxl-runtime-arm64.tar.gz && rm /tmp/voxl-runtime-arm64.tar.gz && exit"

    echo "==> Done. Image loaded on drone."
}

# =========================== DRONE OPERATIONS ================================

cmd_voxl_start() {
    echo "==> Starting drone container on ${VOXL_HOST}..."
    run_remote_compose "up -d"
}

cmd_voxl_shell() {
    echo "==> Connecting to drone container..."
    local running
    running=$(ssh "${VOXL_USER}@${VOXL_HOST}" \
        "docker ps -q -f name=voxl-runtime" 2>/dev/null)

    if [ -z "$running" ]; then
        echo "==> Container not running. Starting it first..."
        cmd_voxl_start
    fi

    ssh -t "${VOXL_USER}@${VOXL_HOST}" \
        "docker exec -it voxl-runtime bash"
}

cmd_voxl_logs() {
    local compose_bin
    compose_bin="$(resolve_remote_compose_bin)" || return 1
    ssh "${VOXL_USER}@${VOXL_HOST}" \
        "cd ${VOXL_DIR} && ${compose_bin} -f docker-compose.yml logs -f --tail=100"
}

cmd_voxl_stop() {
    echo "==> Stopping voxl-drone container..."
    run_remote_compose "down"
}

# =============================================================================
case "${1:-}" in
    setup-qemu)       cmd_setup_qemu ;;
    setup-voxl-services) cmd_setup_voxl_services ;;
    build-deps)       cmd_build_deps ;;
    build-dev)        cmd_build_dev ;;
    build-cross)      cmd_build_cross ;;
    build-runtime)    cmd_build_runtime ;;
    clean-build)      cmd_clean_build ;; 
    dev)              cmd_dev ;;
    cross)            cmd_cross ;;
    build-ws)         shift; cmd_build_ws "$@" ;;
    build-ws-cross)   shift; cmd_build_ws_cross "$@" ;;
    deploy)           cmd_deploy ;;
    export-runtime)   cmd_export_runtime ;;
    deploy-image)     cmd_deploy_image ;;
    voxl-start)       cmd_voxl_start ;;
    voxl-shell)       cmd_voxl_shell ;;
    voxl-logs)        cmd_voxl_logs ;;
    voxl-stop)        cmd_voxl_stop ;;
    help)             help ;;
    *)                help ;;
esac
