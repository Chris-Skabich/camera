import requests
from requests.auth import HTTPDigestAuth
import os
from dotenv import load_dotenv

load_dotenv()
sunapi_url=os.getenv("SUNAPI_URL")
if not sunapi_url:
    print("Error: SUNAPI_URL not found in .env file")
    exit()

uname=os.getenv("USER_NAME")
if not uname:
    print("Error: USER_NAME not found in .env file")
    exit()

password=os.getenv("PASSWORD")
if not password:
    print("Error: PASSWORD not found in .env file")
    exit()

# Load camera info into terminal
response = requests.get(sunapi_url, auth=HTTPDigestAuth(uname, password))

if response.status_code == 200:
    print("Camera Response:\n", response.text)
else:
    print(f"Failed to connect. HTTP Status code: {response.status_code}")