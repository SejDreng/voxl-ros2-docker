# =============================================================================
# Multi-arch, Multi-stage Dockerfile: ROS2 Humble on Ubuntu 22.04
#
# STAGE 1 (voxl-deps)       Third-party libraries and source builds. Rebuild only when a dependency changes. 
#
#
# STAGE 2 (voxl-dev)        Full development environment on top of dependencies: 
#                             - Compilers, debuggers, editors, colcon.
#                           Used on the workstation for development and cross-builds. 
#
#
# STAGE 3 (voxl-runtime)    Minimal runtime: Selected libs from deps, no compilers, no headers. 
#                           Used on the VOXL drone for flight ops
#
# =============================================================================


# =============================================================================
# STAGE 1: DEPENDENCY BASE
# Contains: OS Base -> ROS2 -> apt deps -> source-built libs
# Rate of change: LOW - Should ALMOST never change after all dependencies known
# =============================================================================
FROM ubuntu:22.04 AS voxl-deps

ENV DEBIAN_FRONTEND=noninteractive

# Setup locale
RUN apt-get update && apt-get install -y \
  locales \
  && locale-gen en_US en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && export LANG=en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Set timezone
ENV ROS_VERSION=2
ENV ROS_DISTRO=humble
ENV ROS_PYTHON_VERSION=3
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setup ROS2 Humble Sources
RUN apt-get update && apt-get install -y \
  software-properties-common \
  curl \
  && add-apt-repository universe \
  && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && rm -rf /var/lib/apt/lists/*

# Install ROS2 packages (base + build tools)
#TODO: remove image_transport and image_view dependency from rovtio (not necessary)
RUN apt-get update && apt-get install -y \
  ros-humble-ros-base \
  ros-humble-vision-opencv \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-launch-testing \
  ros-humble-ament-cmake \
  ros-humble-image-transport \
  ros-humble-image-view \
  && rm -rf /var/lib/apt/lists/*

# Install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  wget \
  pkg-config \
  python3-pip \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true && rosdep update --rosdistro humble

# Third-party apt dependencies (dev packages: headers + libs)
RUN echo "===== Building Third-Party apt dependencies ====="
COPY deps/apt-deps.txt /tmp/apt-deps.txt
RUN apt-get update \
  && grep -vE '^\s*#|^\s*$' /tmp/apt-deps.txt | xargs -r apt-get install -y --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

# Third-party Python dependencies
RUN echo "===== Building Third-Party Python dependencies ====="
COPY deps/pip-requirements.txt /tmp/pip-requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/pip-requirements.txt

# Third-party source builds (clone git repos etc... Each build is its own RUN layer so they cache independently - PIN to tag for reproducability)
RUN echo "===== Building Third-Party dependencies ====="
COPY voxl_airlab_thirdparty/ thirdparty/

# CERES-SOLVER/
RUN apt update \
  && cd /thirdparty/build_scripts/ \
  && chmod +x build_ceres.sh \
  && ./build_ceres.sh

# KINDR
RUN apt update \
  && cd /thirdparty/build_scripts/ \
  && chmod +x build_kindr.sh \
  && ./build_kindr.sh 

# ORBSLAM3
RUN apt update \
  && cd /thirdparty/build_scripts/ \
  && chmod +x build_orbslam3.sh \
  && ./build_orbslam3.sh 

# VINS-Core
RUN apt update \
  && cd /thirdparty/build_scripts/ \
  && chmod +x build_vins_core.sh \
  && ./build_vins_core.sh 

# ROSDEP DRYRUN AS LAST STEP OF DEPS SETUP

# Refresh the linker chache so shared libs under /usr/local/lib are found
RUN ldconfig

# =============================================================================
# STAGE 2: FULL DEVELOPMENT ENVIRONMENT
# Interits everything from voxl-deps, adds developer tooling on top
# Rate of change: LOW-MEDIUM - Should rarely change 
# =============================================================================
FROM voxl-deps AS voxl-dev

# Developer tools (not needed to compile project)
RUN apt-get update && apt-get install -y --no-install-recommends \
  gdb \
  vim \
  ssh \
  sudo \
  && rm -rf /var/lib/apt/lists/*

# RMW
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Preemptive directory create
RUN mkdir -p /ros2_ws/build /ros2_ws/install /ros2_ws/log

# Set work dir
WORKDIR /ros2_ws

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
  && echo "if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi" >> /root/.bashrc

CMD ["/bin/bash"]

# =============================================================================
# STAGE 3: VOXL RUNTIME (minimal - no compilers, no headers, no dev tools)
# Starts from a FRESH ubuntu base and picks what is needed to RUN
# Rate of change: LOW - Rebuilt when deps change
# =============================================================================
FROM ubuntu:22.04 AS voxl-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV RC_ALL=C.UTF-8

# Setup ROS2 Humble Sources
RUN apt-get update && apt-get install -y \
  software-properties-common \
  curl \
  && add-apt-repository universe \
  && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && rm -rf /var/lib/apt/lists/*

# Install ROS2 packages (base + build tools) 
#TODO: remove image_transport and image_view dependency from rovtio (not necessary)
RUN apt-get update && apt-get install -y \
  ros-humble-ros-base \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-image-transport \
  ros-humble-image-view \
  && rm -rf /var/lib/apt/lists/*

# Minimal system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3-pip \
  sudo \ 
  && rm -rf /var/lib/apt/lists/*

COPY deps/apt-deps.txt /tmp/apt-deps.txt
RUN apt-get update \
  && grep -vE '^\s*#|^\s*$' /tmp/apt-deps.txt \
    | xargs -r apt-get install -y --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

# Runtime Python dependencies
COPY deps/pip-requirements.txt /tmp/pip-requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/pip-requirements.txt

# Make thirdparty directory (created by COPY??)
# RUN mkdir thirdparty

# Copy source-built libraries from deps stage (Only the installed artifacts - no source code)
COPY --from=voxl-deps /usr/local/lib /usr/local/lib
COPY --from=voxl-deps /usr/local/share/ /usr/local/share
COPY --from=voxl-deps /usr/local/bin /usr/local/bin
COPY --from=voxl-deps /thirdparty /thirdparty
RUN ldconfig

# RMW
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Preemptive directory create
RUN mkdir -p /ros2_ws/build /ros2_ws/install /ros2_ws/log 
WORKDIR /ros2_ws

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
  && echo "if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi" >> /root/.bashrc

CMD ["/bin/bash"]





























# --------------------------OLD!

# FROM ubuntu:22.04 AS voxl-dev

# ENV DEBIAN_FRONTEND=noninteractive

# ENV LANG=C.UTF-8
# ENV LC_ALL=C.UTF-8

# # ---- Core system + build packages ----
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     # Build tools
#     build-essential \
#     cmake \
#     gdb \
#     # Version control & networking
#     git \
#     wget \
#     curl \
#     ssh \
#     # System utilities
#     gnupg2 \
#     lsb-release \
#     software-properties-common \
#     ca-certificates \
#     sudo \
#     # Dev comfort
#     vim \
#     # Python
#     python3 \
#     python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# # ---- Install ROS2 Humble (full: base + build tools) ----
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
#       -o /usr/share/keyrings/ros-archive-keyring.gpg \
#     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
#       http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
#       > /etc/apt/sources.list.d/ros2.list \
#     && apt-get update && apt-get install -y --no-install-recommends \
#       ros-humble-ros-base \
#       ros-humble-rmw-cyclonedds-cpp \
#       ros-humble-launch-testing \
#       ros-humble-ament-cmake \
#       python3-colcon-common-extensions \
#       python3-rosdep \
#       python3-vcstool \
#     && rm -rf /var/lib/apt/lists/*

# # ---- Initialize rosdep ----
# RUN rosdep init || true && rosdep update --rosdistro humble

# # ---- Default RMW and ROS setup ----
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# # ---- Create non-root user ----
# ARG USERNAME=voxl-dev
# ARG USER_UID=1000
# ARG USER_GID=1000
# RUN groupadd --gid ${USER_GID} ${USERNAME} \
#     && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
#     && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME}

# USER ${USERNAME}
# WORKDIR /ros2_ws
# RUN mkdir -p /ros2_ws/build /ros2_ws/install /ros2_ws/log

# RUN echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc \
#     && echo 'if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi' \
#        >> /home/${USERNAME}/.bashrc

# CMD ["/bin/bash"]


# # =============================================================================
# # STAGE 2: VOXL RUNTIME (minimal — no compilers, no dev tools)
# # =============================================================================
# FROM ubuntu:22.04 AS voxl-runtime

# ENV DEBIAN_FRONTEND=noninteractive

# ENV LANG=C.UTF-8
# ENV LC_ALL=C.UTF-8

# # ---- Minimal runtime dependencies only ----
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl \
#     gnupg2 \
#     lsb-release \
#     ca-certificates \
#     python3 \
#     python3-pip \
#     sudo \
#     && rm -rf /var/lib/apt/lists/*

# # ---- Install ROS2 Humble (runtime only — no ament-cmake, no launch-testing) ----
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
#       -o /usr/share/keyrings/ros-archive-keyring.gpg \
#     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
#       http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
#       > /etc/apt/sources.list.d/ros2.list \
#     && apt-get update && apt-get install -y --no-install-recommends \
#       ros-humble-ros-base \
#       ros-humble-rmw-cyclonedds-cpp \
#     && rm -rf /var/lib/apt/lists/*

# # ---- Default RMW and ROS setup ----
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# # ---- Create non-root user ----
# ARG USERNAME=voxl
# ARG USER_UID=1000
# ARG USER_GID=1000
# RUN groupadd --gid ${USER_GID} ${USERNAME} \
#     && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
#     && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME}

# USER ${USERNAME}
# WORKDIR /ros2_ws
# RUN mkdir -p /ros2_ws/build /ros2_ws/install /ros2_ws/log

# RUN echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc \
#     && echo 'if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi' \
#        >> /home/${USERNAME}/.bashrc

# CMD ["/bin/bash"]
