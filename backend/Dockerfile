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

# Copy requirements files (if any are still needed beyond setup.py, like requirements-test.txt)
COPY requirements-test.txt ./
# We'll rely on setup.py for main dependencies via pip install -e .
# If requirements.txt is still needed for other purposes, it can be copied and used separately.
# For now, assuming setup.py is the primary source for runtime deps.

# Install Python build tools
RUN pip install --upgrade pip setuptools wheel
# If requirements-test.txt has content, install it
# Consider RUN pip install -r requirements-test.txt if it contains test-specific deps
# For now, focusing on the main app build

# Install Oh My Zsh (optional)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Copy the package files
COPY . .

# Install the package in development mode (this will use setup.py)
RUN pip install -e .
# If requirements-test.txt has content and wasn't installed earlier, you might install it here
# e.g., RUN pip install -r requirements-test.txt

EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Run Uvicorn with the new package structure
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/backend"]
