import time
import json
import boto3
import requests
import firebase_admin
from firebase_admin import credentials, auth

SECRET_NAME = "heimdall/firebase"
AWS_REGION = "us-east-1"

_token_cache = {
    "token": None,
    "expires_at": 0
}

firebase_initialized = False

def get_secret():
    client = boto3.client(
        "secretsmanager",
        region_name=AWS_REGION
    )
    response = client.get_secret_value(
        SecretId=SECRET_NAME
    )
    return json.loads(response["SecretString"])

def initialize_firebase():
    global firebase_initialized
    if firebase_initialized:
        return
    secret = get_secret()
    cred = credentials.Certificate(
        secret["service_account"]
    )
    firebase_admin.initialize_app(cred)
    firebase_initialized = True

def get_firebase_token():
    initialize_firebase()
    if (
        _token_cache["token"] is None
        or time.time() > _token_cache["expires_at"] - 60
    ):
        print("Renovando token Firebase...")
        secret = get_secret()
        firebase_api_key = secret["FIREBASE_API_KEY"]
        custom_token = auth.create_custom_token(
            "heimdall"
        )
        response = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key={firebase_api_key}",
            json={
                "token": custom_token.decode("utf-8"),
                "returnSecureToken": True
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        _token_cache["token"] = data["idToken"]
        _token_cache["expires_at"] = (
            time.time()
            + int(data.get("expiresIn", 3600))
        )
    return _token_cache["token"]