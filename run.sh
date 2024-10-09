#!/bin/bash

export OPENAI_API_KEY="" # your api key here

echo "Running the application..."
uvicorn env:app --reload & 
UVICORN_PID=$!

python run.py 

kill $UVICORN_PID
