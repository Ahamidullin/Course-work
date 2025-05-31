#!/bin/bash

mkdir -p backend
mkdir -p frontend/src

touch backend/main.py
touch backend/requirements.txt

touch frontend/src/App.jsx
touch frontend/package.json

touch package.json

echo "Done"
tree -L 3
