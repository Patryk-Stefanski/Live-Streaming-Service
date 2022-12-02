from waitress import serve
from livestream import app
import socketio
import logging

hostStr = '0.0.0.0'
portStr = '5000'
threads = 6

logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

# Create socket  and wsgi server
sio = socketio.Server()
server = socketio.WSGIApp(sio, app)


@sio.event
def connect(sid, environ, auth):
    print('connect ', sid)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


if __name__ == '__main__':
    print("Starting Server on {0}:{1}".format( hostStr, portStr))
    print("Threading Enabled for {0} clients".format(threads))
    serve(server, host=hostStr, port=portStr, url_scheme='http', threads=threads, log_untrusted_proxy_headers=True)
