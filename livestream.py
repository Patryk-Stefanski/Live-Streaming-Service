from flask import Flask, render_template, Response, request
import cv2
import time
import numpy as np

global rec
markFace = 0
face = 0
switch = 0

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./FaceDetectionModels/deploy.prototxt.txt',
                               './FaceDetectionModels/res10_300x300_ssd_iter_140000.caffemodel')

# instatiate flask app
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def mark_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        # text = "{:.2f}%".format(confidence * 100)
        text = "W RIZZ % : " + "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    except Exception as e:
        pass
    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if face:
                frame = detect_face(frame)
            if markFace:
                frame = mark_face(frame)

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('face') == 'Face Only':
            global face
            face = not face
            if face:
                time.sleep(4)
        elif request.form.get('markFace') == 'Mark Face':
            global markFace
            markFace = not markFace
            if markFace:
                time.sleep(4)

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
