#!/bin/bash

# Run main.py
echo "Running get_embedding.py"
python get_embedding.py  # Replace with actual command to run main.py

# Run back.py
echo "Running back.py"
python back.py  # Replace with actual command to run back.py

echo "Running check.py"
python check.py

echo "Running concat.py"
python concat.py
echo "Scripts completed."
