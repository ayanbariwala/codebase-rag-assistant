@echo off
echo Starting RAG Assistant...

:: Start backend in a new window
start "RAG Backend" cmd /k "cd backend && venv\Scripts\activate && python main.py"

:: Wait 2 seconds for backend to boot
timeout /t 2 /nobreak > nul

:: Start frontend in a new window  
start "RAG Frontend" cmd /k "cd frontend && python -m http.server 3000"

:: Open the browser automatically
timeout /t 2 /nobreak > nul
start http://localhost:3000

echo Done! Both servers are running.