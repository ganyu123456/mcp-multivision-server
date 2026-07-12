FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY src/ ./src/
COPY pyproject.toml ./

RUN pip install --no-cache-dir -e ".[sse]"

ENV MCP_TRANSPORT=sse \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8093 \
    MCP_VISION_PROVIDER=openai

EXPOSE 8093

ENTRYPOINT ["python", "-m", "mcp_multivision_server.server"]
