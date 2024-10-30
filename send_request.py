import requests

baseUrl = "http://localhost:8080"

def send_request(direction):
    try:
        requests.post(baseUrl + "/set-direction?direction=" + str(direction), "{}")
    except requests.exceptions.ConnectionError: pass