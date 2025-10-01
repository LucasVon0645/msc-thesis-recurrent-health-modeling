# Base image optimized for dev containers
FROM mcr.microsoft.com/devcontainers/base:debian

# Arguments and environment setup
ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode
ARG HOME=/home/$USER
ARG PYTHON_VERSION=3.10.14
ARG R_VERSION=4.3.1

# Install system build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    make \
    ca-certificates \
    software-properties-common \
    gnupg2 \
    lsb-release

# --- Install R from source and required libraries ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libreadline-dev \
    libx11-dev \
    libxt-dev \
    libpng-dev \
    libjpeg-dev \
    libcairo2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libbz2-dev \
    libzstd-dev \
    liblzma-dev \
    libpcre2-dev \
    locales \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://cran.r-project.org/src/base/R-4/R-${R_VERSION}.tar.gz \
    && tar -xf R-${R_VERSION}.tar.gz \
    && cd R-${R_VERSION} \
    && ./configure --with-cairo --with-jpeglib --enable-R-shlib --with-blas --with-lapack \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf R-${R_VERSION}* \
    && locale-gen en_US.UTF-8



# Install essential R packages
RUN Rscript -e "install.packages(c('survival', 'frailtypack', 'survminer'), repos='https://cloud.r-project.org')"

# Revert to non-root user to install Python via pyenv and Poetry
USER $USER
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${HOME}/.local/bin:$PATH"

# --- Install pyenv and Python ---
RUN curl https://pyenv.run | bash \
    && pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION}

# --- Install Poetry ---
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.in-project true

# Default command
CMD [ "poetry", "shell" ]