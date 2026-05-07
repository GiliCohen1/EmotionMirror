# Start both backend and frontend for local development
# Run from the project root: .\dev.ps1

Write-Host "Starting EmotionMirror dev servers..." -ForegroundColor Cyan

# Backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$PSScriptRoot'; Write-Host 'Backend: http://localhost:8000' -ForegroundColor Green; py -m uvicorn backend.main:app --reload --port 8000"

# Frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$PSScriptRoot\frontend'; Write-Host 'Frontend: http://localhost:5173' -ForegroundColor Blue; npm run dev"

Write-Host ""
Write-Host "  Backend  -> http://localhost:8000"   -ForegroundColor Green
Write-Host "  Frontend -> http://localhost:5173"   -ForegroundColor Blue
Write-Host "  API docs -> http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Both servers opened in new windows. Close them to stop." -ForegroundColor Gray
