FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    zsh \
    build-essential \
    curl \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    python3-dev \
    default-libmysqlclient-dev

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Copy requirements files
COPY requirements-test.txt ./
# COPY requirements.txt ./ # requirements.txt is not used for main deps, setup.py is

# Install Python build tools
RUN pip install --upgrade pip setuptools wheel

# Install test dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-test.txt

# Install Oh My Zsh for a better shell experience (optional, consider removing if not needed)
# RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Copy the rest of the application code
COPY . .

# Install the application and its dependencies with cache mount
# This will pick up dependencies from setup.py's install_requires
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

# Set the Python path
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8000

# Default command to run the application (can be overridden in docker-compose.yml)
# Note: The main docker-compose.yml overrides this.
CMD ["sh", "-c", "export PYTHONPATH=/app && python -m uvicorn backend.langchain_mysql:app --host 0.0.0.0 --port 8000"]
