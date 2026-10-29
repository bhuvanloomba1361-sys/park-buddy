from flask import Flask, Response, jsonify, request, send_from_directory
import cv2
import json
import os
import threading
import time
import numpy as np
from parking_detector import ParkingDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLOTS_FILE = os.path.join(BASE_DIR, 'config', 'slots.json')
app = Flask(__name__, static_folder='static')
detector = ParkingDetector()
lock = threading.Lock()
inference_lock = threading.Lock()
latest_frame = None
latest_jpeg = None
latest_status = {'vehicles_detected': 0, 'slots_total': 0, 'slots_occupied': 0, 'slots_available': 0, 'updated_at': None}


def load_slots():
    try:
        with open(SLOTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def save_slots(slots):
    os.makedirs(os.path.dirname(SLOTS_FILE), exist_ok=True)
    with open(SLOTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(slots, f, indent=2)


@app.get('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')



@app.get('/api/health')
def health():
    return jsonify({'ok': True, 'model': 'yolov8n.pt', 'message': 'Smart Parking server is running'})


@app.get('/api/status')
def status():
    with lock:
        return jsonify(latest_status)


@app.get('/api/slots')
def get_slots():
    return jsonify(load_slots())


@app.post('/api/slots')
def update_slots():
    data = request.get_json(silent=True) or {}
    slots = data.get('slots', data if isinstance(data, list) else None)
    if not isinstance(slots, list):
        return jsonify({'error': 'Expected a JSON list of parking slots'}), 400
    cleaned = []
    for i, slot in enumerate(slots):
        try:
            x, y, w, h = [int(slot[k]) for k in ('x', 'y', 'w', 'h')]
            if w <= 0 or h <= 0:
                continue
            cleaned.append({'id': str(slot.get('id', f'S{i+1}')), 'x': x, 'y': y, 'w': w, 'h': h})
        except (KeyError, TypeError, ValueError):
            continue
    save_slots(cleaned)
    # Refresh the dashboard's slot counters immediately, without waiting for another frame.
    with lock:
        occupied_ids = set(latest_status.get('occupied_slot_ids', []))
        valid_occupied = [str(slot.get('id', 'Slot')) for slot in cleaned if str(slot.get('id', 'Slot')) in occupied_ids]
        latest_status.update({
            'slots_total': len(cleaned),
            'slots_occupied': len(valid_occupied),
            'slots_available': max(0, len(cleaned) - len(valid_occupied)),
            'occupied_slot_ids': valid_occupied,
        })
    return jsonify({'ok': True, 'slots': cleaned, 'status': latest_status})


@app.post('/api/frame')
def receive_frame():
    global latest_frame, latest_jpeg, latest_status
    if 'frame' not in request.files:
        return jsonify({'error': 'Upload a frame using form field named frame'}), 400
    raw = request.files['frame'].read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400
    slots = load_slots()
    # Serialize inference so simultaneous browser requests cannot stall the model.
    with inference_lock:
        annotated, count, occupied_ids = detector.process(frame, slots)
    total = len(slots)
    ok, encoded = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        with lock:
            latest_frame = annotated
            latest_jpeg = encoded.tobytes()
            latest_status = {
                'vehicles_detected': count,
                'slots_total': total,
                'slots_occupied': len(occupied_ids),
                'slots_available': max(0, total - len(occupied_ids)),
                'occupied_slot_ids': occupied_ids,
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    return jsonify({'ok': True, 'status': latest_status})


@app.get('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                frame = latest_jpeg
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.08)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.get('/user')
def user_page():
    return send_from_directory(app.static_folder, 'user.html')


if __name__ == '__main__':
    print('Smart Parking AI running at http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=False, threaded=True)
