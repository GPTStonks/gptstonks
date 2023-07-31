FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Update system
RUN apt update && apt install -y python3-pip python3-venv

# Configure Poetry
ENV POETRY_VERSION=1.5.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /openbb-chat/

# Add dependencies
ADD poetry.lock /openbb-chat/poetry.lock
ADD pyproject.toml /openbb-chat/pyproject.toml

# Install fastapi and uvicorn for running the fastapi app
RUN poetry add fastapi uvicorn

RUN poetry install --no-root
ADD openbb_chat /openbb-chat/openbb_chat
ADD README.md /openbb-chat/README.md
RUN poetry install --only-root

#Install graphical tool to avoid interaction
RUN apt install -y libwebkit2gtk-4.0-dev

# Add other files
ADD scripts /openbb-chat/scripts/
ADD data /openbb-chat/data/

# Create app directory
RUN mkdir -p /openbb-chat/fastapi

# Copy FastAPI app
COPY ./fastapi /openbb-chat/fastapi

WORKDIR /openbb-chat/fastapi

# Expose port for FastAPI app to run on
EXPOSE 8000

# Run the FastAPI app
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
