import requests

baseUrl = "http://localhost"
port = 80

def send_request(direction):
    try:
        requests.post(f"{baseUrl}:{port}/set-direction?direction={direction}", "{}")
    except requests.exceptions.ConnectionError: pass

def set_port(new_port):
    global port
    port = new_port