"""Security helpers: URL validation (SSRF protection), credential redaction, and optional API token auth."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse, urlunparse

from fastapi import Header, HTTPException, Query
from .config import settings

_CREDENTIAL_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)([^/@]+@)")


def redact_url_credentials(url: str) -> str:
    """Mask userinfo credentials embedded in a stream URL before logging or returning."""
    if not url:
        return ""
    if url.startswith(("demo://", "webcam:")):
        return url
    return _CREDENTIAL_RE.sub(r"\1***:***@", url)


def _is_forbidden_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address, allow_private: bool) -> bool:
    """Return True if an IP address violates network SSRF policy."""
    if ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified:
        return True
    if not allow_private and (ip_obj.is_private or ip_obj.is_loopback):
        return True
    return False


def validate_stream_source(
    raw_url: str,
    *,
    allow_private: bool | None = None,
    allowed_hosts: str | None = None,
) -> str:
    """Validate a camera stream source against SSRF and scheme policies."""
    url = raw_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Stream URL cannot be empty.")

    lower = url.lower()
    if lower in {"demo://crowd-sim", "demo://panic-sim", "demo://crush-sim"}:
        return lower

    if lower.startswith("webcam:"):
        idx_str = lower.split(":", 1)[1]
        if not idx_str.isdigit() or int(idx_str) > 9:
            raise HTTPException(status_code=400, detail="Webcam index must be webcam:0 through webcam:9.")
        return lower

    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"rtsp", "rtsps", "http", "https"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported stream scheme. Use rtsp://, rtsps://, http://, https://, webcam:0, or demo://crowd-sim.",
        )
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="Stream URL must include a valid hostname or IP address.")

    host = parsed.hostname.strip().lower()
    allow_priv = settings.allow_private_rtsp if allow_private is None else allow_private
    hosts_cfg = settings.rtsp_allowed_hosts if allowed_hosts is None else allowed_hosts
    allowlist = {h.strip().lower() for h in hosts_cfg.split(",") if h.strip()}

    if allowlist and host not in allowlist:
        raise HTTPException(status_code=403, detail=f"Camera host '{host}' is not in RTSP_ALLOWED_HOSTS.")

    if host in {"metadata.google.internal", "169.254.169.254"}:
        raise HTTPException(status_code=403, detail="Access to link-local metadata endpoints is forbidden.")

    try:
        ip_obj = ipaddress.ip_address(host)
        if _is_forbidden_ip(ip_obj, allow_priv):
            raise HTTPException(status_code=403, detail=f"Target IP '{host}' blocked by SSRF network policy.")
    except ValueError:
        # Hostname rather than IP literal; check resolution if private IPs are disallowed.
        if host == "localhost" and not allow_priv:
            raise HTTPException(status_code=403, detail="Loopback camera URLs are disabled by policy.")
        if not allow_priv:
            try:
                infos = socket.getaddrinfo(host, parsed.port or 554, proto=socket.IPPROTO_TCP)
                for info in infos:
                    resolved_ip = ipaddress.ip_address(info[4][0])
                    if _is_forbidden_ip(resolved_ip, allow_priv):
                        raise HTTPException(
                            status_code=403,
                            detail=f"Resolved address for '{host}' is blocked by SSRF network policy.",
                        )
            except socket.gaierror:
                # Allow OpenCV to handle unreachable hostname cleanly if DNS fails locally
                pass

    return url


def mask_parsed_url(url: str) -> str:
    """Return URL with credentials redacted using urllib parsing when possible."""
    if not url or url.startswith(("demo://", "webcam:")):
        return url
    try:
        p = urlparse(url)
        if p.username or p.password:
            host_part = p.hostname or ""
            if p.port:
                host_part = f"{host_part}:{p.port}"
            netloc = f"***:***@{host_part}"
            return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        pass
    return redact_url_credentials(url)


def require_auth(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    """Enforce bearer / API-key auth only when AUTH_ENABLED=true."""
    if not settings.auth_enabled:
        return
    expected = settings.api_auth_token.strip()
    if not expected:
        raise HTTPException(status_code=500, detail="AUTH_ENABLED=true requires API_AUTH_TOKEN to be configured.")
    provided = None
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    elif x_api_key:
        provided = x_api_key.strip()
    elif token:
        provided = token.strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token.")

