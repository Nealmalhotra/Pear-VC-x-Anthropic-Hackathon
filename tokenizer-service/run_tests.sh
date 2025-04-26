#!/bin/bash

# Change to the script directory
cd "$(dirname "$0")"

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run the tests
echo "Running unit tests..."
python3 -m unittest discover -s tests

echo "Tests completed." 