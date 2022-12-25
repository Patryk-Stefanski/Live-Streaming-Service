from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('faceDetectionModels/haarcascade_frontalface_default.xml')

# instantiate flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Load trained face model from recognizer folder
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainer.yml")

# open labels file generated during face training, so we can use the labels for face recognition
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# initialize camera and set fps cap
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 60)


def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)  # detects faces in gray frame

    for (x, y, w, h) in faces:
        aoi = gray[y:y + h, x:x + w]  # get the area of interest(frame surrounding the detected face)

        id_, conf = recognizer.predict(aoi)  # compare trained model to detected face
        if 5 <= conf <= 85:  # only recognize people if the confidence level is between 5(high) and 85(low)
            name = labels[id_]
            txt = "{0} Confidence: {1}".format(name, round(conf))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # mark face recognition with border
            cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if conf > 85:
            name = "Unknown"
            txt = "{0} Confidence: {1}".format(name, round(conf))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # mark face recognition with border
            cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)



    return frame


def gen_frames():  # generate frame by frame from camera
    count = 0
    while True:
        count += 1  # count frames
        # If camera read is ok , detect faces and render the image
        success, frame = camera.read()
        if success:
            if count % 2 == 0:  # generate every second frame
                frame = recognize_face(frame)  # call the face detect function above

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(e)

        if cv2.waitKey(1) == ord('q'):
            break

    # Turn off camera and close windows
    camera.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    try:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(e)


def run():
    socketio.run(app, threaded=True, processes=3)


if __name__ == '__main__':
    try:
        socketio.run(app)
    except Exception as excp:
        print(excp)