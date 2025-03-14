FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    zsh \
    build-essential \
    default-libmysqlclient-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Install Oh My Zsh (optional)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

COPY . .

EXPOSE 8000

# Run Uvicorn without reload
CMD ["python", "-m", "uvicorn", "backend.langchain_mysql:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
