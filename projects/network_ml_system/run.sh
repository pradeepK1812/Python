
#!/bin/bash

# Start server in background
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
PID=$!

echo "Server started with PID $PID"

# Handle CTRL+C (SIGINT)
trap "echo 'Stopping server...'; kill $PID; wait $PID; echo 'Clean exit'; exit 0" SIGINT SIGTERM

# Wait for process
wait $PID
