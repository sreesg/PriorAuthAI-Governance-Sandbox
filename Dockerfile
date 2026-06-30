FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py agent_engine.py challenger_agent.py s3_helper.py ./
COPY generate_evidence_docs.py ./
COPY agent.js app.js cases.js hooks.js skills.js regoInterpreter.js ./
COPY index.html index.css help.html main.html ./
COPY avi-icon.svg ./
COPY rules_declaration.md skills_declaration.md rules.rego ./
COPY extracted_policies.json generated_skills.json ./
COPY policies/ ./policies/
COPY skills/ ./skills/
COPY cases/ ./cases/

# Copy Clinical Reasoning Fabric source
COPY src/ ./src/

# Copy CRF frontend panels to serve as static JS
COPY src/clinical_reasoning_fabric/frontend/*.js ./static/crf/

# Add src to Python path (instead of pip install -e)
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/agent/policies || exit 1

# Start server
CMD ["python3", "server.py"]
