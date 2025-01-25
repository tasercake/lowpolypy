FROM python:3.13-slim-bookworm as builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a virtual environment
RUN uv venv
# Set the virtual environment as the default python environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy requirements files
COPY pyproject.toml uv.lock /app/
# Copy the other app files
COPY src /app/src

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv uv sync --group server

EXPOSE 8000

CMD ["uvicorn", "lowpolypy.server:app", "--host", "0.0.0.0", "--port", "8000"]
