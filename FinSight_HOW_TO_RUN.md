cd "C:\Users\Tausif\Desktop\FinSight DS project"
.\.venv\Scripts\Activate.ps1
uvicorn api.app:app --reload --port 8000




in the second terminal do these
cd "C:\Users\Tausif\Desktop\FinSight DS project"
python -m http.server 5500 --directory dashboard


then open 
http://127.0.0.1:5500