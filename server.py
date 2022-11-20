from waitress import serve
from livestream import app
import socketio
import logging

hostStr = '0.0.0.0'
portStr = '5000'

logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

# Create socket  and wsgi server
sio = socketio.Server()
server = socketio.WSGIApp(sio, app)

if __name__ == '__main__':
    serve(server, host=hostStr, port=portStr, url_scheme='http', threads=6, expose_tracebacks=True,
          log_untrusted_proxy_headers=True)
