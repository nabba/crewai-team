"""
Host Bridge Service — Controlled external resource access for containerized agents.

Runs NATIVELY on macOS host (not in Docker). Agents in Docker connect via
host.docker.internal:9100. Same pattern as signal-cli host service.

Every external interaction passes through explicit capability gating:
  - Per-agent capability tokens with scoped permissions
  - Path/host allowlists enforced at infrastructure level
  - 4-tier risk escalation (LOW → MEDIUM → HIGH → CRITICAL)
  - CRITICAL actions require human approval via Signal
  - Every action audit-logged to ~/.crewai-bridge/audit.jsonl
  - File-based kill switch: touch ~/.crewai-bridge/KILL

Safety invariant: Self-Improver's allowed_paths NEVER include SOUL.md,
philosophical RAG, capability config, or bridge code. Enforced by path
exclusion in capabilities.json, not by agent self-restraint.

Start: python host_bridge/main.py
Or:    launchctl load host_bridge/com.crewai.bridge.plist
"""

import json
import logging
import os
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("host_bridge")

# ── Configuration ─────────────────────────────────────────────────────────────

BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "9100"))
BRIDGE_DIR = Path.home() / ".crewai-bridge"
AUDIT_LOG_PATH = BRIDGE_DIR / "audit.jsonl"
CAPABILITY_STORE_PATH = BRIDGE_DIR / "capabilities.json"
KILL_SWITCH_PATH = BRIDGE_DIR / "KILL"

# Fallback: look for capabilities.json next to this script
REPO_CAPABILITY_PATH = Path(__file__).parent / "capabilities.json"

BRIDGE_DIR.mkdir(parents=True, exist_ok=True)


# ── Risk Tiers ────────────────────────────────────────────────────────────────

class RiskTier(str, Enum):
    LOW = "low"           # Read-only, no side effects
    MEDIUM = "medium"     # Write operations, bounded scope
    HIGH = "high"         # Process execution, network mutation
    CRITICAL = "critical" # Requires human approval via Signal

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


# ── Capability Model ──────────────────────────────────────────────────────────

class Capability(BaseModel):
    agent_id: str
    token: str
    allowed_actions: list[str]       # e.g. ["filesystem.read", "network.http"]
    allowed_paths: list[str] = []    # glob patterns for filesystem
    blocked_paths: list[str] = []    # explicit exclusions (safety)
    allowed_hosts: list[str] = []    # for network requests
    max_requests_per_minute: int = 30
    risk_ceiling: str = "medium"     # max risk tier without escalation


# ── State ─────────────────────────────────────────────────────────────────────

_capabilities: dict[str, Capability] = {}
_rate_tracker: dict[str, list[float]] = {}
_kill_switch = False
_pending_approvals: dict[str, Optional[bool]] = {}  # approval_id → True/False/None


# ── Capability loading ────────────────────────────────────────────────────────

def load_capabilities():
    global _capabilities
    # Try home dir first, then repo dir
    for path in [CAPABILITY_STORE_PATH, REPO_CAPABILITY_PATH]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for cap_data in data:
                    cap = Capability(**cap_data)
                    _capabilities[cap.token] = cap
                logger.info(f"Loaded {len(_capabilities)} capability tokens from {path}")
                return
            except Exception as e:
                logger.error(f"Failed to load capabilities from {path}: {e}")
    logger.warning("No capabilities.json found — bridge will reject all requests")


def get_capability(token: str) -> Capability:
    cap = _capabilities.get(token)
    if not cap:
        raise HTTPException(status_code=403, detail="Invalid capability token")
    return cap


def check_permission(cap: Capability, action: str):
    if action not in cap.allowed_actions and "*" not in cap.allowed_actions:
        raise HTTPException(
            status_code=403,
            detail=f"Agent '{cap.agent_id}' not authorized for: {action}"
        )


def check_risk_tier(cap: Capability, required_tier: str):
    if RISK_ORDER.get(cap.risk_ceiling, 0) < RISK_ORDER.get(required_tier, 0):
        raise HTTPException(
            status_code=403,
            detail=f"Action requires {required_tier} risk tier; agent has {cap.risk_ceiling}"
        )


def check_path(cap: Capability, target_path: str):
    """Validate path against allowed_paths and blocked_paths."""
    resolved = str(Path(target_path).resolve())

    # Blocked paths always win (safety-critical)
    for blocked in cap.blocked_paths:
        if resolved.startswith(blocked.rstrip("*")) or Path(resolved).match(blocked):
            raise HTTPException(status_code=403, detail=f"Path explicitly blocked for agent '{cap.agent_id}'")

    # Check allowed paths
    if not cap.allowed_paths:
        raise HTTPException(status_code=403, detail="No filesystem paths allowed")

    for allowed in cap.allowed_paths:
        if resolved.startswith(allowed.rstrip("*")) or Path(resolved).match(allowed):
            return

    raise HTTPException(status_code=403, detail=f"Path not in allowed paths for agent '{cap.agent_id}'")


# ── Rate Limiter ──────────────────────────────────────────────────────────────

def check_rate_limit(cap: Capability):
    now = time.time()
    key = cap.token
    timestamps = _rate_tracker.get(key, [])
    timestamps = [t for t in timestamps if now - t < 60.0]
    if len(timestamps) >= cap.max_requests_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    timestamps.append(now)
    _rate_tracker[key] = timestamps


# ── Kill Switch ───────────────────────────────────────────────────────────────

def check_kill_switch():
    global _kill_switch
    if KILL_SWITCH_PATH.exists():
        _kill_switch = True
    if _kill_switch:
        raise HTTPException(status_code=503, detail="Kill switch activated — all operations halted")


# ── Audit Logger ──────────────────────────────────────────────────────────────

def audit_log(agent_id: str, action: str, details: dict, result: str):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "action": action,
        "details": {k: str(v)[:500] for k, v in details.items()},
        "result": result,
    }
    try:
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
    logger.info(f"AUDIT: {agent_id}/{action} → {result}")


# ── Human Approval via Signal ─────────────────────────────────────────────────

def request_human_approval(agent_id: str, action: str, details: str) -> bool:
    """Send approval request via signal-cli, poll for response."""
    approval_id = str(uuid.uuid4())[:8]
    _pending_approvals[approval_id] = None

    phone = os.getenv("SIGNAL_OWNER_NUMBER", "")
    if not phone:
        logger.warning("No SIGNAL_OWNER_NUMBER set — cannot request approval")
        return False

    message = (
        f"🤖 BRIDGE APPROVAL [{approval_id}]\n"
        f"Agent: {agent_id}\n"
        f"Action: {action}\n"
        f"Details: {details[:500]}\n\n"
        f"Reply '{approval_id} yes' to approve\n"
        f"Reply '{approval_id} no' to deny"
    )

    try:
        signal_http = os.getenv("SIGNAL_HTTP_URL", "http://localhost:7583")
        bot_number = os.getenv("SIGNAL_BOT_NUMBER", "")
        import httpx
        httpx.post(f"{signal_http}/v2/send", json={
            "message": message,
            "number": bot_number,
            "recipients": [phone],
        }, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send Signal approval request: {e}")
        return False

    # Poll for response (5 minute timeout)
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(5)
        result = _pending_approvals.get(approval_id)
        if result is not None:
            del _pending_approvals[approval_id]
            return result

    _pending_approvals.pop(approval_id, None)
    return False  # Timeout = deny


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="CrewAI Host Bridge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def authenticate(x_capability_token: str = Header(...)) -> Capability:
    check_kill_switch()
    cap = get_capability(x_capability_token)
    check_rate_limit(cap)
    return cap


@app.on_event("startup")
def startup():
    load_capabilities()
    logger.info(f"Host Bridge starting on port {BRIDGE_PORT}")
    logger.info(f"Kill switch path: {KILL_SWITCH_PATH}")
    logger.info(f"Audit log: {AUDIT_LOG_PATH}")


# ── Filesystem Endpoints ──────────────────────────────────────────────────────

class FileReadRequest(BaseModel):
    path: str
    encoding: str = "utf-8"
    max_bytes: int = Field(default=1_000_000, le=10_000_000)

class FileWriteRequest(BaseModel):
    path: str
    content: str
    encoding: str = "utf-8"
    create_dirs: bool = False

class FileListRequest(BaseModel):
    path: str
    pattern: str = "*"
    recursive: bool = False

@app.post("/filesystem/read")
def fs_read(req: FileReadRequest, cap: Capability = Depends(authenticate)):
    check_permission(cap, "filesystem.read")
    check_path(cap, req.path)

    target = Path(req.path).resolve()
    if not target.exists():
        audit_log(cap.agent_id, "filesystem.read", {"path": req.path}, "NOT_FOUND")
        raise HTTPException(status_code=404, detail="File not found")

    try:
        content = target.read_text(encoding=req.encoding)[:req.max_bytes]
    except Exception as e:
        audit_log(cap.agent_id, "filesystem.read", {"path": req.path}, f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])

    audit_log(cap.agent_id, "filesystem.read", {"path": req.path, "size": len(content)}, "OK")
    return {"content": content, "size": len(content), "path": str(target)}

@app.post("/filesystem/write")
def fs_write(req: FileWriteRequest, cap: Capability = Depends(authenticate)):
    check_permission(cap, "filesystem.write")
    check_risk_tier(cap, "medium")
    check_path(cap, req.path)

    target = Path(req.path).resolve()
    if req.create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(req.content, encoding=req.encoding)
    audit_log(cap.agent_id, "filesystem.write", {"path": req.path, "size": len(req.content)}, "OK")
    return {"written": len(req.content), "path": str(target)}

@app.post("/filesystem/list")
def fs_list(req: FileListRequest, cap: Capability = Depends(authenticate)):
    check_permission(cap, "filesystem.list")
    check_path(cap, req.path)

    target = Path(req.path).resolve()
    if req.recursive:
        files = [str(f) for f in target.rglob(req.pattern)]
    else:
        files = [str(f) for f in target.glob(req.pattern)]

    audit_log(cap.agent_id, "filesystem.list", {"path": req.path, "count": len(files)}, "OK")
    return {"files": files[:1000], "count": len(files)}

# ── Network Endpoints ─────────────────────────────────────────────────────────

class HttpRequest(BaseModel):
    method: str = "GET"
    url: str
    headers: dict = {}
    body: Optional[str] = None
    timeout: int = Field(default=30, le=120)

@app.post("/network/http")
def net_http(req: HttpRequest, cap: Capability = Depends(authenticate)):
    check_permission(cap, "network.http")

    from urllib.parse import urlparse
    host = urlparse(req.url).hostname
    if cap.allowed_hosts and host not in cap.allowed_hosts and "*" not in cap.allowed_hosts:
        audit_log(cap.agent_id, "network.http", {"url": req.url}, "DENIED_HOST")
        raise HTTPException(status_code=403, detail=f"Host '{host}' not in allowed hosts")

    import httpx
    response = httpx.request(
        method=req.method, url=req.url,
        headers=req.headers, content=req.body,
        timeout=req.timeout,
    )

    audit_log(cap.agent_id, "network.http", {
        "method": req.method, "url": req.url, "status": response.status_code
    }, "OK")
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": response.text[:100_000],
    }

class NetworkScanRequest(BaseModel):
    subnet: str = "192.168.1.0/24"
    ports: list[int] = [80, 443, 8080, 22]
    timeout: float = 2.0

@app.post("/network/scan")
def net_scan(req: NetworkScanRequest, cap: Capability = Depends(authenticate)):
    check_permission(cap, "network.scan")
    check_risk_tier(cap, "high")

    import ipaddress
    results = []
    network = ipaddress.IPv4Network(req.subnet, strict=False)

    for ip in list(network.hosts())[:256]:
        ip_str = str(ip)
        for port in req.ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(req.timeout)
                if sock.connect_ex((ip_str, port)) == 0:
                    results.append({"ip": ip_str, "port": port, "status": "open"})
                sock.close()
            except Exception:
                pass

    audit_log(cap.agent_id, "network.scan", {"subnet": req.subnet, "found": len(results)}, "OK")
    return {"results": results, "scanned_subnet": req.subnet}

# ── Process Execution ─────────────────────────────────────────────────────────

# IMMUTABLE: commands that are always blocked
BLOCKED_COMMANDS = frozenset({
    "rm -rf /", "mkfs", "dd if=/dev", "shutdown", "reboot",
    ":(){ :|:& };:", "chmod -R 777 /", "format",
})

class ExecuteRequest(BaseModel):
    command: list[str]
    working_dir: str = "/tmp"
    timeout: int = Field(default=30, le=300)
    env: dict = {}

@app.post("/execute")
def execute(req: ExecuteRequest, cap: Capability = Depends(authenticate)):
    check_permission(cap, "execute")
    check_risk_tier(cap, "high")

    cmd_str = " ".join(req.command)
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_str:
            audit_log(cap.agent_id, "execute", {"cmd": req.command}, "BLOCKED")
            raise HTTPException(status_code=403, detail=f"Command blocked by safety filter")

    # CRITICAL risk tier → require human approval
    if RISK_ORDER.get(cap.risk_ceiling, 0) >= RISK_ORDER.get("critical", 3):
        approved = request_human_approval(
            cap.agent_id, "execute",
            f"Command: {cmd_str}\nDir: {req.working_dir}"
        )
        if not approved:
            audit_log(cap.agent_id, "execute", {"cmd": req.command}, "DENIED_BY_HUMAN")
            raise HTTPException(status_code=403, detail="Human approval denied or timed out")

    try:
        result = subprocess.run(
            req.command, capture_output=True, text=True,
            timeout=req.timeout, cwd=req.working_dir,
            env={**os.environ, **req.env},
        )
        audit_log(cap.agent_id, "execute", {
            "cmd": req.command, "returncode": result.returncode
        }, "OK")
        return {
            "stdout": result.stdout[:50_000],
            "stderr": result.stderr[:10_000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        audit_log(cap.agent_id, "execute", {"cmd": req.command}, "TIMEOUT")
        raise HTTPException(status_code=408, detail="Command timed out")

# ── GPU/Ollama Proxy ──────────────────────────────────────────────────────────

class GpuInferenceRequest(BaseModel):
    model: str = "qwen3:30b-a3b"
    prompt: str
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048

@app.post("/gpu/inference")
def gpu_inference(req: GpuInferenceRequest, cap: Capability = Depends(authenticate)):
    check_permission(cap, "gpu.inference")

    import httpx
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    response = httpx.post(
        f"{ollama_url}/api/generate",
        json={
            "model": req.model,
            "prompt": req.prompt,
            "system": req.system,
            "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
            "stream": False,
        },
        timeout=120,
    )

    data = response.json()
    audit_log(cap.agent_id, "gpu.inference", {
        "model": req.model, "prompt_len": len(req.prompt)
    }, "OK")
    return {"response": data.get("response", ""), "model": req.model}

# ── Approval webhook (for Signal responses) ───────────────────────────────────

class ApprovalResponse(BaseModel):
    approval_id: str
    approved: bool

@app.post("/approval/respond")
def approval_respond(req: ApprovalResponse):
    """Called by signal forwarder when owner replies to an approval request."""
    if req.approval_id in _pending_approvals:
        _pending_approvals[req.approval_id] = req.approved
        return {"status": "recorded"}
    return {"status": "unknown_approval_id"}

# ── Health / Status ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if not _kill_switch else "killed",
        "kill_switch": _kill_switch,
        "capabilities_loaded": len(_capabilities),
        "pending_approvals": len([v for v in _pending_approvals.values() if v is None]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/status")
def status(cap: Capability = Depends(authenticate)):
    check_permission(cap, "status")
    import platform
    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "bridge_port": BRIDGE_PORT,
    }

# ── Kill switch management ────────────────────────────────────────────────────

@app.post("/kill")
def activate_kill():
    """Emergency kill — no auth required (localhost only)."""
    global _kill_switch
    _kill_switch = True
    KILL_SWITCH_PATH.touch()
    audit_log("system", "kill_switch", {}, "ACTIVATED")
    return {"status": "kill_switch_activated"}

@app.post("/unkill")
def deactivate_kill(cap: Capability = Depends(authenticate)):
    """Restore from kill switch. Requires commander token."""
    global _kill_switch
    if cap.agent_id != "commander" and cap.risk_ceiling != "critical":
        raise HTTPException(status_code=403, detail="Only commander can unkill")
    _kill_switch = False
    KILL_SWITCH_PATH.unlink(missing_ok=True)
    audit_log(cap.agent_id, "kill_switch", {}, "DEACTIVATED")
    return {"status": "kill_switch_deactivated"}

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BRIDGE_PORT)
