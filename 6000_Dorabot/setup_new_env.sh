#!/bin/bash
set -e

echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Updating pip..."
python3 -m pip install --upgrade pip

echo "Installing core dependencies..."
pip install networkx shapely pygame cmd2 enum34 numpy pandas matplotlib

echo "Installing protobuf and grpcio-tools..."
# We install unrestricted protobuf so it gets the latest (e.g. 5.x or 4.x)
pip install protobuf grpcio-tools

echo "Regenerating protobuf files..."
python3 -m grpc_tools.protoc -I ./src/protocol --python_out=./src/protocol ./src/protocol/data_book.proto

echo "Checking if pybox2d needs to be installed..."
if ! python3 -c "import Box2D" &> /dev/null; then
    echo "pybox2d not found. Attempting installation from source..."
    echo "Note: This requires 'swig' space. (brew install swig on Mac / sudo apt install swig on Linux)"
    if [ ! -d "pybox2d" ]; then
        git clone https://github.com/pybox2d/pybox2d
    fi
    cd pybox2d || exit
    pip install .
    cd ..
else
    echo "pybox2d is already installed."
fi

echo "====================================="
echo "Environment setup complete!"
echo "Use 'source venv/bin/activate' to activate."
echo "You can test the simulator via 'cd src && python simulator.py -t 1'"
