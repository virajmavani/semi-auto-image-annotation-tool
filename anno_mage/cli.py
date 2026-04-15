import threading
import webbrowser
import uvicorn


def main():
    port = 8000
    threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run("web.backend.main:app", host="127.0.0.1", port=port)
