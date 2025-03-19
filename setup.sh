#!/bin/bash
echo "🔧 Setting up the project..."
pip install -r requirements.txt
docker-compose up --build -d
