#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Codellama 7b instruct
python3 -m llama_cpp.server --model $MODEL_PATH --n_gpu_layers $N_GPU_LAYERS --port $PORT --n_ctx $N_CTX
