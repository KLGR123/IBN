#!/bin/bash

export OPENAI_API_KEY="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"

echo "Running the application..."
uvicorn env:app --reload & 
UVICORN_PID=$!

python run.py 

kill $UVICORN_PID
