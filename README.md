# Smart Parking AI (YOLO + Flask)

A beginner-friendly demo that detects vehicles in live camera frames with Ultralytics YOLO and estimates whether configured parking rectangles are occupied.

## Requirements
- Windows 10/11 (or macOS/Linux)
- Python 3.10–3.12, 64-bit recommended
- Internet connection for the first install and first YOLO model download

## Run on Windows
1. Extract this ZIP file.
2. Open the extracted `smart_parking_yolo_project` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   ```
   If `py -3.11` is unavailable, try `python -m venv .venv`.
5. Activate it (PowerShell):
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```
   Alternatively, skip activation and run commands using `\.venv\Scripts\python.exe`.
6. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Start the app:
   ```powershell
   python app.py
   ```
8. Open `http://127.0.0.1:5000` on the laptop. The first run downloads `yolov8n.pt` automatically.

## Use a phone camera
For browser camera access from a phone to a laptop, browsers generally require HTTPS on remote addresses. For testing, install Cloudflare Tunnel (`cloudflared`) and run this in a second terminal while the Flask server remains running:

```powershell
cloudflared tunnel --url http://localhost:5000
```

Open the generated `https://....trycloudflare.com` address on your phone. Keep both terminals open. A Quick Tunnel URL is temporary; do not use it to expose a sensitive or production system. You can also use a compatible IP-camera app and configure its stream separately if you extend the app.

Cloudflare downloads: https://developers.cloudflare.com/tunnel/downloads/
Quick Tunnel documentation: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/

## Configure parking spaces
In the web dashboard, enter each slot's ID and rectangle coordinates (`x`, `y`, `width`, `height`) relative to the camera image, then click **Add slot** and **Save slots**. Start the camera. Green rectangles show vehicle detections; configured slot rectangles are colored red when estimated occupied and blue/orange when available. If no slots are configured, the dashboard can count vehicles but cannot calculate parking-space availability.

## API
- `GET /api/health` — health check
- `GET /api/status` — latest counts
- `GET /api/slots` — configured slot rectangles
- `POST /api/slots` — save slot list as JSON, e.g. `{"slots":[{"id":"S1","x":50,"y":50,"w":100,"h":100}]}`
- `POST /api/frame` — upload an image in multipart form field `frame`
- `GET /video_feed` — annotated MJPEG stream from the most recently processed frame

## Notes
- This is a demo, not a certified parking management system.
- Slot occupancy is estimated by checking whether the center of a detected vehicle box falls inside a slot rectangle; configure the rectangles for your camera angle.
- Model inference can be slow on older laptops. Start with a small camera resolution and close other heavy applications.
- The first installation may require several minutes and an internet connection to download Python packages and the YOLO model.
