
#!/bin/bash

#server env variables
export HOST=0.0.0.0
export PORT=8000

# Ollama configuration
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:1b

# linear model config 
export LINEAR_MODEL_PATH=ml/linear_regression/model.pt
export API_VERSION=v1

#logistic model config
export LOGISTIC_MODEL_PATH=ml/logistic_regression/model.pt

# ------------------------------------------------------------------
# Verify Ollama installation.
# ------------------------------------------------------------------

if ! command -v ollama >/dev/null 2>&1
then
    echo "Error: Ollama is not installed."
    echo "Run ./.devcontainer/install_ollama.sh"
    exit 1
fi
# ------------------------------------------------------------------
# Ensure Ollama server is running.
# ------------------------------------------------------------------

if ! curl -s "$OLLAMA_HOST/api/tags" >/dev/null 2>&1
then
    echo "Starting Ollama..."

    ollama serve &


    echo "Waiting for Ollama..."

    until curl -s $OLLAMA_HOST/api/tags >/dev/null 2>&1
    do
        sleep 1
    done

    echo "Ollama is ready."
fi

# Start server in background
uvicorn app.api:app --host $HOST --port $PORT --log-level info --log-config logging.yaml  &

FASTAPI_PID=$!

echo "Server started with PID $FASTAPI_PID"

# Handle CTRL+C (SIGINT)
trap "echo 'Stopping server...'; kill $FASTAPI_PID; wait $FASTAPI_PID; echo 'Clean exit'; exit 0" SIGINT SIGTERM

# Wait for process
wait $FASTAPI_PID
