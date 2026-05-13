
#!/bin/bash

#server env variables
export HOST=0.0.0.0
export PORT=8000

#model config 
export LINEAR_MODEL_PATH=ml/linear_regression/model.pt
export API_VERSION=v1


# Start server in background
uvicorn app.api:app --host $HOST --port $PORT --log-level info --log-config logging.yaml  &

PID=$!

echo "Server started with PID $PID"

# Handle CTRL+C (SIGINT)
trap "echo 'Stopping server...'; kill $PID; wait $PID; echo 'Clean exit'; exit 0" SIGINT SIGTERM

# Wait for process
wait $PID
