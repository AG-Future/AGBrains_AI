import requests

baseUrl = "http://localhost:8080"

def send_request(direction):
    requests.post(baseUrl + "/set-direction?direction=" + str(direction), "{}")