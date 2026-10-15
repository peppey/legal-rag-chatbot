import datetime
import time

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient
from google.auth import crypt, jwt

from legal_rag_chatbot import api, auth

KID = "test-key"


@pytest.fixture(scope="module")
def signing_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    return crypt.RSASigner.from_string(private_pem, KID), cert_pem


@pytest.fixture
def make_token(monkeypatch: pytest.MonkeyPatch, signing_key):
    signer, cert_pem = signing_key
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.setattr(auth, "_signing_certs", lambda refresh=False: {KID: cert_pem})

    def make(**overrides) -> str:
        now = int(time.time())
        claims = {
            "iss": auth.ISSUER,
            "aud": auth.PROJECT_ID,
            "sub": "uid-1",
            "iat": now,
            "exp": now + 3600,
            "role": "a",
        }
        claims.update(overrides)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(signer, claims, header={"kid": KID}).decode()

    return make


def _status(token: str | None, path: str = "/api/documents/x/file") -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return TestClient(api.app).get(path, headers=headers).status_code


def test_a_valid_token_grants_the_groups_of_its_role(make_token) -> None:
    request = type("R", (), {"headers": {"Authorization": f"Bearer {make_token()}"}})
    access = auth.current_access(request)  # type: ignore[arg-type]

    assert access.user_id == "uid-1"
    assert access.groups == {"A", "Geteilt"}


def test_valid_tokens_pass_and_invalid_ones_are_rejected(make_token) -> None:
    # 404: authenticated, but the document does not exist
    assert _status(make_token()) == 404
    assert _status(None) == 401
    assert _status("not-a-token") == 401
    assert _status(make_token(exp=int(time.time()) - 3600)) == 401
    assert _status(make_token(aud="other-project")) == 401
    assert _status(make_token(iss="https://evil.example")) == 401
    assert _status(make_token(sub=None)) == 401


def test_a_token_signed_with_another_key_is_rejected(
    make_token, monkeypatch: pytest.MonkeyPatch, signing_key
) -> None:
    token = make_token()
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "other")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(other.public_key())
        .serial_number(2)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(other, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    monkeypatch.setattr(auth, "_signing_certs", lambda refresh=False: {KID: pem})

    assert _status(token) == 401


@pytest.mark.parametrize("role", [None, "", "root", "ADMIN", 5])
def test_tokens_without_a_known_role_are_forbidden(make_token, role) -> None:
    assert _status(make_token(role=role)) == 403


def test_the_dev_header_is_ignored_outside_the_dev_mode(make_token) -> None:
    response = TestClient(api.app).get(
        "/api/documents/x/file", headers={"X-Dev-Role": "admin"}
    )
    assert response.status_code == 401
