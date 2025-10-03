# Base image optimized for dev containers
FROM mcr.microsoft.com/devcontainers/base:debian

# Arguments and environment setup
ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode
ARG HOME=/home/$USER
ARG PYTHON_VERSION=3.10.14

# System build tools & libs needed for building Python with pyenv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    make \
    wget \
    llvm \
    # Python build deps
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    # helpful/required for some tooling
    libxml2-dev \
    libxmlsec1-dev \
    software-properties-common \
    gnupg2 \
    lsb-release \
    # use system python3 only to bootstrap Poetry installer
    python3 \
    python3-venv \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Switch to non-root user for pyenv & Poetry
USER $USER
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${HOME}/.local/bin:$PATH"

# Install pyenv and desired Python
RUN curl https://pyenv.run | bash \
 && pyenv install ${PYTHON_VERSION} \
 && pyenv global ${PYTHON_VERSION}

# Install Poetry (creates in ~/.local/bin)
RUN curl -sSL https://install.python-poetry.org | python3 - \
 && poetry config virtualenvs.in-project true

# Default command
CMD [ "poetry", "shell" ]
