# =============================================================================
# Makefile — Wrapper for build.sh commands
# Usage: make <target>
# =============================================================================

SHELL := /usr/bin/env bash
SCRIPT := ./scripts/build.sh

# ---- HELP ----

.PHONY: help
help: ## Display all commands
	@$(SCRIPT) help

# ---- ONE-TIME SETUP ----

.PHONY: setup-qemu
setup-qemu: ## Install QEMU user-static for arm64 emulation (run once)
	@$(SCRIPT) setup-qemu

# ---- BUILD IMAGES ----

.PHONY: build-deps
build-deps: ## Build only the dependency base stage (useful to verify deps)
	@$(SCRIPT) build-deps

.PHONY: build-dev
build-dev: ## Build the full dev image (native x86_64)
	@$(SCRIPT) build-dev

.PHONY: build-cross
build-cross: ## Build the full dev image for arm64 via QEMU
	@$(SCRIPT) build-cross

.PHONY: build-runtime
build-runtime: ## Build the slim runtime image for arm64
	@$(SCRIPT) build-runtime

.PHONY: clean-build
clean-build: # Erase build images, containers, builders, and artifacts to allow for a complete rebuild
	@$(SCRIPT) clean-build

# ---- DEVELOPMENT (workstation) ----

.PHONY: dev
dev: ## Open a shell in the native x86 dev container
	@$(SCRIPT) dev

.PHONY: cross
cross: ## Open a shell in the arm64 QEMU dev container
	@$(SCRIPT) cross

.PHONY: build-ws
build-ws: ## Run colcon build in the native dev container
	@$(SCRIPT) build-ws

.PHONY: build-ws-cross
build-ws-cross: ## Run colcon build in the arm64 container (produces arm64 binaries)
	@$(SCRIPT) build-ws-cross

# ---- DEPLOY TO DRONE ----

.PHONY: export-runtime
export-runtime: ## Save the slim runtime image to a .tar.gz file
	@$(SCRIPT) export-runtime

.PHONY: extract-install
extract-install: ## Copy cross-built arm64 install/ out of the Docker volume
	@$(SCRIPT) extract-install

.PHONY: deploy
deploy: ## Rsync source + install + compose to drone
	@$(SCRIPT) deploy

.PHONY: deploy-image
deploy-image: ## Transfer the runtime image .tar.gz to drone and load it
	@$(SCRIPT) deploy-image

# ---- DRONE OPERATIONS (via SSH) ----

.PHONY: voxl-start
voxl-start: ## Start the voxl-drone container
	@$(SCRIPT) voxl-start

.PHONY: voxl-shell
voxl-shell: ## Attach to the running voxl-drone container
	@$(SCRIPT) voxl-shell

.PHONY: voxl-logs
voxl-logs: ## Show voxl-drone container logs
	@$(SCRIPT) voxl-logs

.PHONY: voxl-stop
voxl-stop: ## Stop the voxl-drone container
	@$(SCRIPT) voxl-stop

# ---- COMPOUND TARGETS ----

.PHONY: build-all-images
build-all-images: build-dev build-cross build-runtime ## Build all images

.PHONY: deploy-all
deploy-all: build-runtime export-runtime deploy deploy-image ## Full build + deploy pipeline

.DEFAULT_GOAL := help