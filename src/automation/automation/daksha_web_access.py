"""Shared browser-access gate for Daksha web UIs.

Reads config/web_access.yaml to decide whether each UI requires the Daksha
app's launch-key auth (false) or allows any browser (true).

Flask apps (one call)
---------------------
    from daksha_web_access import setup_flask_gate
    setup_flask_gate(app, "vr_management")

FastAPI / Starlette apps
------------------------
    from daksha_web_access import is_access_enabled
    # The FastAPI gate reads BROWSER_ACCESS from os.environ.

http.server apps
----------------
    from daksha_web_access import guard_http_handler
    guard_http_handler(HandlerClass, "data_collection")
"""

import logging
import os

import yaml

_HERE = os.path.dirname(os.path.realpath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "config", "web_access.yaml")
_WEBPAGE_DIR = os.path.normpath(os.path.join(_HERE, "..", "webpage"))
if not os.path.isdir(_WEBPAGE_DIR):
    # Running from install/: the env files live in the package share dir.
    try:
        from ament_index_python.packages import get_package_share_directory
        _WEBPAGE_DIR = os.path.join(get_package_share_directory("automation"), "webpage")
    except Exception:
        pass

log = logging.getLogger("daksha.web_access")


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            doc = yaml.safe_load(f) or {}
        return doc.get("browser_access", {})
    except FileNotFoundError:
        log.warning("web_access.yaml not found at %s — defaulting to allow all", _CONFIG_PATH)
        return {}
    except Exception:
        log.exception("Failed to read %s — defaulting to allow all", _CONFIG_PATH)
        return {}


def is_access_enabled(app_name: str) -> bool:
    """Return True if *app_name* allows any browser (gate off)."""
    config = _load_config()
    enabled = config.get(app_name, True)
    if not enabled:
        log.info("%s: gate ACTIVE (launch-key auth required)", app_name)
    return bool(enabled)


def _load_gate_env() -> None:
    """Load webpage/daksha.conf and webpage/.env into os.environ (setdefault)."""
    for name in ("daksha.conf", ".env"):
        path = os.path.join(_WEBPAGE_DIR, name)
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
        except FileNotFoundError:
            pass


def setup_flask_gate(flask_app, app_name: str) -> None:
    """Load gate secrets and register the Daksha gate on a Flask app.

    When web_access.yaml says *app_name* is true, the gate is off and any
    browser can open the UI. When false, only the Daksha Flutter app (with
    its launch key) can authenticate.
    """
    from daksha_flask_gate import FlaskGateSettings, register_flask_gate

    _load_gate_env()
    if is_access_enabled(app_name):
        os.environ["BROWSER_ACCESS"] = "true"
    else:
        os.environ.pop("BROWSER_ACCESS", None)
    register_flask_gate(flask_app, FlaskGateSettings.from_env())
    os.environ.pop("BROWSER_ACCESS", None)


def guard_http_handler(handler_class, app_name: str):
    """Register the Daksha gate on a BaseHTTPRequestHandler subclass.

    When web_access.yaml says *app_name* is true, the handler is returned
    unchanged. When false, only the Daksha Flutter app (with its launch key)
    can authenticate, using the same /_gate/* protocol and signed session
    cookies as the Flask gate.
    """
    if is_access_enabled(app_name):
        return handler_class

    import hmac
    import json
    import secrets
    import time
    from http.cookies import CookieError, SimpleCookie
    from urllib.parse import parse_qs, urlparse

    from daksha_flask_gate import (
        GATE_PREFIX, _MAX_FAILURES, FlaskGateSettings, _Tickets,
        _make_session_token, _safe_next, _verify_session_token,
    )

    _load_gate_env()
    os.environ.pop("BROWSER_ACCESS", None)
    settings = FlaskGateSettings.from_env()
    if not settings.launch_keys:
        raise RuntimeError("APP_LAUNCH_KEY must hold at least one key")

    tickets = _Tickets(settings.session_secret)
    failures: dict[str, list[float]] = {}
    revoked: set[str] = set()

    def _send(self, status: int, body: bytes = b"", headers: tuple = ()):
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _session_cookie(self):
        cookies = SimpleCookie()
        try:
            cookies.load(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        morsel = cookies.get(settings.cookie_name)
        return morsel.value if morsel else None

    def _has_valid_session(self) -> bool:
        token = _session_cookie(self)
        if not token or token in revoked:
            return False
        return _verify_session_token(settings.session_secret, token)

    def _cookie_header(value: str, max_age: int) -> str:
        parts = [f"{settings.cookie_name}={value}", f"Max-Age={max_age}",
                 "Path=/", "HttpOnly", "SameSite=Lax"]
        if settings.cookie_secure:
            parts.append("Secure")
        return "; ".join(parts)

    def _create_ticket(self):
        ip = self.client_address[0] if self.client_address else "unknown"
        now = time.monotonic()
        recent = [t for t in failures.get(ip, []) if now - t < 60]
        if recent:
            failures[ip] = recent
        else:
            failures.pop(ip, None)
        if len(recent) >= _MAX_FAILURES:
            return _send(self, 429)

        key = self.headers.get("x-app-key", "").encode()
        if not sum(hmac.compare_digest(key, k) for k in settings.launch_keys):
            failures.setdefault(ip, []).append(time.monotonic())
            log.warning("rejected ticket request from %s", ip)
            return _send(self, 401)

        ticket = secrets.token_urlsafe(32)
        tickets.add(ticket, settings.ticket_ttl)
        body = json.dumps({"ticket": ticket, "expires_in": settings.ticket_ttl}).encode()
        _send(self, 200, body, (("Content-Type", "application/json"),
                                ("Cache-Control", "no-store")))

    def _entry(self, query: dict):
        ticket = query.get("ticket", [""])[0]
        if not ticket or not tickets.pop(ticket):
            log.info("rejected ticket exchange from %s", self.client_address[0])
            return _send(self, 403)

        old = _session_cookie(self)
        if old:
            revoked.add(old)

        session = _make_session_token(settings.session_secret, settings.session_ttl)
        target = _safe_next(query.get("next", ["/"])[0])
        _send(self, 303, headers=(
            ("Location", target),
            ("Set-Cookie", _cookie_header(session, settings.session_ttl)),
            ("Cache-Control", "no-store"),
            ("Referrer-Policy", "no-referrer"),
        ))

    def _logout(self):
        token = _session_cookie(self)
        if token:
            revoked.add(token)
        _send(self, 204, headers=(("Set-Cookie", _cookie_header("", 0)),))

    def _handle_gate(self, method: str) -> bool:
        """Serve /_gate/* or block requests without a session; True if handled."""
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith(GATE_PREFIX + "/"):
            if method == "POST":
                # Drain the body so the connection stays usable.
                self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            route = (method, path[len(GATE_PREFIX):])
            if route == ("POST", "/ticket"):
                _create_ticket(self)
            elif route == ("GET", "/entry"):
                _entry(self, parse_qs(parsed.query))
            elif route == ("GET", "/session"):
                _send(self, 204 if _has_valid_session(self) else 401)
            elif route == ("POST", "/logout"):
                _logout(self)
            else:
                _send(self, 404)
            return True
        if any(path.startswith(p) for p in settings.public_paths):
            return False
        if _has_valid_session(self):
            return False
        wants_page = method == "GET" and "text/html" in self.headers.get("accept", "")
        _send(self, 404 if wants_page else 401)
        return True

    original_get = getattr(handler_class, "do_GET", None)
    original_post = getattr(handler_class, "do_POST", None)

    def do_GET(self):
        if not _handle_gate(self, "GET") and original_get:
            original_get(self)

    def do_POST(self):
        if not _handle_gate(self, "POST"):
            if original_post:
                original_post(self)
            else:
                _send(self, 501)

    handler_class.do_GET = do_GET
    handler_class.do_POST = do_POST
    log.info("%s: Daksha gate registered", app_name)
    return handler_class
