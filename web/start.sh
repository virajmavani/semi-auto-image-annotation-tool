#!/bin/bash

echo "🚀 Starting Anno-Mage Web Application"
echo "======================================"

# Check if backend virtual environment exists
# if [ ! -d "backend/venv" ]; then
#     echo "⚠️  Backend virtual environment not found!"
#     echo "Please run setup first:"
#     echo "  cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
#     exit 1
# fi

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "⚠️  Frontend dependencies not installed!"
    echo "Please run setup first:"
    echo "  cd frontend && npm install"
    exit 1
fi

echo ""
echo "Starting Backend Server..."
cd backend
# source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

echo "Waiting for backend to start..."
sleep 3

echo ""
echo "Starting Frontend Development Server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both servers are starting!"
echo ""
echo "Backend API:  http://localhost:8000"
echo "Frontend App: http://localhost:3000"
echo "API Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
