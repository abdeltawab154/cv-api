# face_stream_fastapi.py

import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI()

# Initialize camera and face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangles on detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def root():
    html_content = """
    <html>
        <head>
            <title>Face Detection Stream</title>
        </head>
        <body>
            <h1>Face detection stream is running</h1>
            <img src="/video" width="640" height="480">
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run("face_stream_fastapi:app", host="0.0.0.0", port=5000)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
