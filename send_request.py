import requests

baseUrl = "http://192.168.1.19:8080"
'''
"http://192.168.1.90:8080"
'''
def send_request(direction):
    requests.post(baseUrl + "/set-direction?direction=" + str(direction), "{}")