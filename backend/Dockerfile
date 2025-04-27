FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    zsh \
    build-essential \
    default-libmysqlclient-dev \
    default-mysql-client \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

COPY requirements.txt requirements-test.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt -r requirements-test.txt

# Install Oh My Zsh (optional)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Copy the package files
COPY . .

# Install the package in development mode
RUN pip install -e .

EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src

# Run Uvicorn with the new package structure
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src/backend"]
