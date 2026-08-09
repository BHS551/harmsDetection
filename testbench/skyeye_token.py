"""Emite un ID token de Firebase para un uid dado, usando la cuenta de servicio
de heimdall/firebase (mismo mecanismo que usa el worker en firebase_auth.py)."""
import json
import boto3
import requests
import firebase_admin
from firebase_admin import credentials, auth

SECRET_NAME = "heimdall/firebase"
AWS_REGION = "us-east-1"

_secret = None
_app = None


def get_secret():
    global _secret
    if _secret is None:
        sm = boto3.client("secretsmanager", region_name=AWS_REGION)
        _secret = json.loads(sm.get_secret_value(SecretId=SECRET_NAME)["SecretString"])
    return _secret


def get_id_token(uid, claims=None):
    global _app
    secret = get_secret()
    if _app is None:
        _app = firebase_admin.initialize_app(
            credentials.Certificate(secret["service_account"])
        )
    custom = auth.create_custom_token(uid, claims or {})
    r = requests.post(
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
        f"?key={secret['FIREBASE_API_KEY']}",
        json={"token": custom.decode("utf-8"), "returnSecureToken": True},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["idToken"]


if __name__ == "__main__":
    import sys

    uid = sys.argv[1] if len(sys.argv) > 1 else "skyeye-test-harness"
    t = get_id_token(uid)
    print(f"ID token obtenido para {uid}: {len(t)} chars, prefijo {t[:12]}...")
