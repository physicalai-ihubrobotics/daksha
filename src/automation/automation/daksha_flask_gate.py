"""Daksha gate for Flask web UIs: same /_gate/* protocol as daksha_gate.py.

Registers /_gate/ticket, /_gate/entry, /_gate/session and /_gate/logout on a
Flask app and adds a before_request hook that blocks unauthenticated browsers
when BROWSER_ACCESS is not true.

Usage (three lines, same pattern as the FastAPI gate)::

    from daksha_flask_gate import FlaskGateSettings, register_flask_gate
    settings = FlaskGateSettings.from_env()
    register_flask_gate(app, settings)

Session cookies are **signed tokens**: ``expiry_unix.nonce.hmac``. Any service
that shares the same SESSION_SECRET can verify them without a shared store, so
a session created on port 8100 works on 8101, 8102, etc. (browsers send
cookies to every port on the same host).
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass

from flask import Blueprint, Response, jsonify, make_response, redirect, request

log = logging.getLogger("daksha.flask_gate")

GATE_PREFIX = "/_gate"
_MAX_FAILURES = 10


def _env_secret(name: str) -> str:
    value = os.environ.get(name, "")
    if len(value) < 32:
        raise RuntimeError(
            f"{name} must be a random value of at least 32 characters. "
            "Copy .env.example to .env and fill it in."
        )
    return value


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return default if not value else value not in ("0", "false", "no")


def _env_list(name: str, default: str = "") -> tuple:
    return tuple(v.strip() for v in os.environ.get(name, default).split(",") if v.strip())


@dataclass(frozen=True)
class FlaskGateSettings:
    session_secret: bytes
    launch_keys: tuple
    cookie_name: str = "daksha_session"
    cookie_secure: bool = True
    ticket_ttl: int = 60
    session_ttl: int = 8 * 3600
    public_paths: tuple = ()
    browser_access: bool = False

    @classmethod
    def from_env(cls) -> "FlaskGateSettings":
        if _env_flag("BROWSER_ACCESS", False):
            return cls(session_secret=b"", launch_keys=(), browser_access=True)
        return cls(
            session_secret=_env_secret("SESSION_SECRET").encode(),
            launch_keys=tuple(k.encode() for k in _env_list("APP_LAUNCH_KEY")),
            cookie_name=os.environ.get("COOKIE_NAME", "daksha_session"),
            cookie_secure=_env_flag("COOKIE_SECURE", True),
            ticket_ttl=int(os.environ.get("TICKET_TTL", "60")),
            session_ttl=int(os.environ.get("SESSION_TTL", str(8 * 3600))),
            public_paths=_env_list("PUBLIC_PATHS"),
        )


# -- Tickets: random, in-memory, one-time-use (single process) ---------------

class _Tickets:
    def __init__(self, secret: bytes) -> None:
        self._secret = secret
        self._expiry: dict[str, float] = {}

    def _digest(self, token: str) -> str:
        return hmac.new(self._secret, token.encode(), hashlib.sha256).hexdigest()

    def add(self, token: str, ttl: int) -> None:
        now = time.monotonic()
        self._expiry = {k: t for k, t in self._expiry.items() if t > now}
        self._expiry[self._digest(token)] = now + ttl

    def pop(self, token: str) -> bool:
        return self._expiry.pop(self._digest(token), 0) > time.monotonic()


# -- Sessions: signed tokens, verifiable by any process with the secret -------

def _make_session_token(secret: bytes, ttl: int) -> str:
    """Create a signed session token: ``expiry.nonce.signature``."""
    expiry = int(time.time()) + ttl
    nonce = secrets.token_urlsafe(16)
    payload = f"{expiry}.{nonce}".encode()
    sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return f"{expiry}.{nonce}.{sig}"


def _verify_session_token(secret: bytes, token: str) -> bool:
    """Verify a signed session token without any lookup."""
    parts = token.split(".", 2)
    if len(parts) != 3:
        return False
    expiry_str, nonce, sig = parts
    try:
        expiry = int(expiry_str)
    except ValueError:
        return False
    if time.time() > expiry:
        return False
    payload = f"{expiry_str}.{nonce}".encode()
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def _safe_next(path: str) -> str:
    if not path.startswith("/") or path.startswith("//") or "\\" in path:
        return "/"
    return path


def register_flask_gate(flask_app, settings: FlaskGateSettings) -> None:
    """Wire the Daksha gate into a Flask app."""

    if settings.browser_access:
        log.warning("BROWSER_ACCESS is on: any browser can open this web UI")
        return

    if not settings.launch_keys:
        raise RuntimeError("APP_LAUNCH_KEY must hold at least one key")

    tickets = _Tickets(settings.session_secret)
    failures: dict[str, list[float]] = {}
    revoked: set[str] = set()

    gate = Blueprint("daksha_gate", __name__, url_prefix=GATE_PREFIX)

    @gate.route("/ticket", methods=["POST"])
    def create_ticket():
        ip = request.remote_addr or "unknown"
        now = time.monotonic()
        recent = [t for t in failures.get(ip, []) if now - t < 60]
        if recent:
            failures[ip] = recent
        else:
            failures.pop(ip, None)
        if len(recent) >= _MAX_FAILURES:
            return Response(status=429)

        key = request.headers.get("x-app-key", "").encode()
        if not sum(hmac.compare_digest(key, k) for k in settings.launch_keys):
            failures.setdefault(ip, []).append(time.monotonic())
            log.warning("rejected ticket request from %s", ip)
            return Response(status=401)

        ticket = secrets.token_urlsafe(32)
        tickets.add(ticket, settings.ticket_ttl)
        resp = jsonify({"ticket": ticket, "expires_in": settings.ticket_ttl})
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @gate.route("/entry", methods=["GET"])
    def entry():
        ticket = request.args.get("ticket", "")
        if not ticket or not tickets.pop(ticket):
            log.info("rejected ticket exchange from %s", request.remote_addr)
            return Response(status=403)

        old = request.cookies.get(settings.cookie_name)
        if old:
            revoked.add(old)

        session = _make_session_token(settings.session_secret, settings.session_ttl)
        target = _safe_next(request.args.get("next", "/"))
        resp = make_response(redirect(target, code=303))
        resp.set_cookie(
            settings.cookie_name,
            session,
            max_age=settings.session_ttl,
            httponly=True,
            secure=settings.cookie_secure,
            samesite="Lax",
            path="/",
        )
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    def _has_valid_session() -> bool:
        token = request.cookies.get(settings.cookie_name)
        if not token or token in revoked:
            return False
        return _verify_session_token(settings.session_secret, token)

    @gate.route("/session", methods=["GET"])
    def check_session():
        if _has_valid_session():
            return Response(status=204)
        return Response(status=401)

    @gate.route("/logout", methods=["POST"])
    def logout():
        token = request.cookies.get(settings.cookie_name)
        if token:
            revoked.add(token)
        resp = Response(status=204)
        resp.delete_cookie(
            settings.cookie_name,
            path="/",
            secure=settings.cookie_secure,
            httponly=True,
            samesite="Lax",
        )
        return resp

    flask_app.register_blueprint(gate)

    def _is_public(path: str) -> bool:
        return any(path.startswith(p) for p in settings.public_paths)

    @flask_app.before_request
    def _require_session():
        path = request.path
        if path.startswith(GATE_PREFIX + "/"):
            return None
        if _is_public(path):
            return None
        if _has_valid_session():
            return None
        wants_page = request.method in ("GET", "HEAD") and "text/html" in request.headers.get("accept", "")
        return Response(status=404 if wants_page else 401)
