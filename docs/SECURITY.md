# Security and Privacy Checklist

- **Credential Redaction**: Stream URLs containing embedded `username:password@` credentials are masked (`***:***@`) before saving to SQLite or returning via API endpoints (`app/security.py`).
- **SSRF Mitigations**: `validate_stream_source` rejects invalid schemes, cloud metadata / link-local addresses (`169.254.0.0/16`), and supports `RTSP_ALLOWED_HOSTS` allowlisting plus `ALLOW_PRIVATE_RTSP=false` for hardened deployments.
- **API Authentication**: Enable `AUTH_ENABLED=true` and set a strong `API_AUTH_TOKEN` in `backend/.env` for any shared or venue network deployment. Always terminate TLS via reverse proxy (Nginx / Caddy / Traefik) in production.
- **Secret Hygiene**: Never put credentials or `API_AUTH_TOKEN` in `frontend/.env` `VITE_*` variables.
- **Upload Limits**: File formats (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) and `MAX_UPLOAD_MB` are validated before processing.
- **Retention & Authorization**: Configure retention cleanup for `processed/` and `data/crowdguard.db` according to venue CCTV privacy policy, and obtain authorization before ingesting real camera feeds.
