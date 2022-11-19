import eventlet
import socketio
from waitress import serve

from livestream import app

sio = socketio.Server()
server = socketio.WSGIApp(sio, app)

if __name__ == '__main__':
    serve(server, host='0.0.0.0', port=5000, url_scheme='http', threads=6, expose_tracebacks=True,
          log_untrusted_proxy_headers=True)
