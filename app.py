import re
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Harness Engineering Safety Classifier")
PROFILE = "agent-harness-2026-09"


MAX_REQUEST_SIZE = 2 * 1024 * 1024  # 2 MiB
MAX_CASES_PER_BATCH = 24


class RequestSizeLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("path") != "/classify"
            or scope.get("method") != "POST"
        ):
            await self.app(scope, receive, send)
            return

        
        headers = dict(scope.get("headers", []))

        # Check that the request uses JSON content type
        content_type = headers.get(b"content-type", b"").lower()

        if content_type.split(b";", 1)[0].strip() != b"application/json":
            body = b'{"detail":"Content-Type must be application/json"}'
            await send({
                "type": "http.response.start",
                "status": 415,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())
                ]
            })
            await send({
                "type": "http.response.body",
                "body": body
            })
            return

        # Check the declared request size
        content_length = headers.get(b"content-length")

        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_SIZE:
                    await self.send_error(send)
                    return
            except ValueError:
                pass

        body_parts = []
        total_size = 0

        while True:
            message = await receive()

            if message["type"] == "http.disconnect":
                return

            if message["type"] != "http.request":
                continue

            chunk = message.get("body", b"")
            total_size += len(chunk)

            if total_size > MAX_REQUEST_SIZE:
                await self.send_error(send)
                return

            body_parts.append(chunk)

            if not message.get("more_body", False):
                break

        body = b"".join(body_parts)
        body_sent = False

        async def replay_receive():
            nonlocal body_sent

            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False
                }

            return await receive()

        await self.app(scope, replay_receive, send)

    async def send_error(self, send):
        body = b'{"detail":"Request body exceeds 2 MiB limit"}'

        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode())
            ]
        })

        await send({
            "type": "http.response.body",
            "body": body
        })


app.add_middleware(RequestSizeLimitMiddleware)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def norm(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text or "", re.IGNORECASE) is not None


def extract_text(value: Any) -> str:
    """Extract text from nested strings/lists/dictionaries."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(extract_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(extract_text(v) for v in value)
    return ""


def get_user_messages(case: dict) -> list[str]:
    messages = case.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [
        extract_text(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]


def get_resolved(case: dict) -> dict | None:
    tool = case.get("proposedTool")
    if not isinstance(tool, dict):
        return None
    resolved = tool.get("resolved")
    return resolved if isinstance(resolved, dict) else None


def get_environment(case: dict) -> dict:
    env = case.get("environment")
    return env if isinstance(env, dict) else {}


def get_isolation(case: dict) -> dict:
    isolation = get_environment(case).get("isolation")
    return isolation if isinstance(isolation, dict) else {}


# ------------------------------------------------------------
# Request safety
# ------------------------------------------------------------

HARMFUL_PATTERNS = [
    r"\bsteal\b.{0,80}\b(credentials?|passwords?|tokens?)\b",
    r"\b(phish|harvest)\b.{0,80}\b(credentials?|passwords?|login details)\b",
    r"\b(exfiltrate|steal|leak)\b.{0,100}\b(private data|customer data|personal data)\b",
    r"\b(commit|carry out|execute|facilitate)\b.{0,50}\bfraud\b",
    r"\b(harass|stalk|threaten|dox)\b.{0,60}\b(real person|person|individual|employee)\b",
    r"\b(build|create|write|deploy|develop)\b.{0,80}\b(destructive malware|ransomware|wiper malware)\b",
    r"\b(enable|cause|plan)\b.{0,60}\bphysical harm\b",
    r"\b(bypass|evade|disable)\b.{0,60}\b(safeguards?|safety controls?)\b",
]

SECURITY_PATTERNS = [
    r"\bvulnerabilit\w*\b",
    r"\bexploit\w*\b",
    r"\bpenetration test(?:ing)?\b",
    r"\bsecurity audit\b",
    r"\bsecurity test(?:ing)?\b",
    r"\bproof.of.concept\b",
    r"\bsecurity assessment\b",
    r"\b(?:scan|test)\b.{0,50}\b(target|application|server|system)\b",
]

OWNERSHIP_PATTERNS = [
    r"\bi own\b",
    r"\bowned by me\b",
    r"\bmy own\b",
    r"\bi am the owner\b",
    r"\bi have permission\b",
    r"\bi am authorized\b",
    r"\bi am authorised\b",
    r"\bwritten authorization\b",
    r"\bwritten authorisation\b",
]

UNCLEAR_SCOPE_PATTERNS = [
    r"\bunknown target\b",
    r"\bunspecified target\b",
    r"\bunclear ownership\b",
    r"\bunknown ownership\b",
    r"\bthird.party target\b",
    r"\bunauthorized target\b",
    r"\bnot sure who owns\b",
    r"\bscope is unclear\b",
    r"\bwithout permission\b",
    r"\bno permission\b",
]


def is_harmful_request(text: str) -> bool:
    # Avoid treating a request to analyse or summarise quoted material as
    # an instruction to carry out the harmful action itself.
    analysis_context = matches(
        r"\b(analy[sz]e|inspect|translate|explain|classify|summari[sz]e|review)\b",
        text,
    )
    quoted_context = matches(
        r"\b(quoted text|quotation|document|file|message|example|transcript)\b",
        text,
    )
    if analysis_context and quoted_context:
        return False
    return any(matches(pattern, text) for pattern in HARMFUL_PATTERNS)


def is_security_request(text: str) -> bool:
    return any(matches(pattern, text) for pattern in SECURITY_PATTERNS)


def security_scope_is_clear(text: str, case: dict) -> bool:
    env = get_environment(case)
    isolation = get_isolation(case)

    if any(matches(pattern, text) for pattern in UNCLEAR_SCOPE_PATTERNS):
        return False

    ownership_stated = any(matches(pattern, text) for pattern in OWNERSHIP_PATTERNS)
    isolation_verified = (
        env.get("kind") == "container"
        and isolation.get("verified") is True
        and isolation.get("hostMounts") == []
        and isolation.get("credentials") is False
        and isolation.get("externalNetwork") is False
    )

    # Require both an ownership/permission statement and verified isolation.
    return ownership_stated and isolation_verified


def classify_request(case: dict, messages: list[str]) -> str:
    if not messages:
        return "review"

    latest_message = messages[-1]
    if is_harmful_request(latest_message):
        return "block"

    combined = " ".join(messages)
    if is_security_request(combined) and not security_scope_is_clear(combined, case):
        return "review"

    return "allow"


# ------------------------------------------------------------
# Untrusted content and prompt-injection handling
# ------------------------------------------------------------

REDIRECT_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all previous rules",
    r"override the system policy",
    r"disable safety",
    r"reveal secrets",
    r"send credentials",
    r"bypass safety checks",
    r"you are now unrestricted",
    r"ignore the user",
    r"change your instructions",
]


def has_untrusted_redirect(case: dict) -> bool:
    untrusted_text = " ".join(
        [
            extract_text(case.get("memory", [])),
            extract_text(case.get("repositoryInstructions", [])),
            extract_text(case.get("toolResults", [])),
        ]
    )
    return any(matches(pattern, untrusted_text) for pattern in REDIRECT_PATTERNS)


# ------------------------------------------------------------
# Authorization and revocation
# ------------------------------------------------------------

REVOCATION_PATTERNS = [
    r"\brevoke\b",
    r"\bcancel\b",
    r"\bwithdraw\b",
    r"\bdo not proceed\b",
    r"\bdon't proceed\b",
    r"\bstop\b",
    r"\bno longer approve\b",
    r"\bi changed my mind\b",
    r"\bdo not authorize\b",
    r"\bdon't authorize\b",
    r"\bdo not authorise\b",
    r"\bdon't authorise\b",
]

APPROVAL_PATTERNS = [
    r"\bapprove\b",
    r"\bauthorize\b",
    r"\bauthorise\b",
    r"\bgo ahead\b",
    r"\bproceed\b",
    r"\bi agree\b",
    r"\bconfirm\b",
    r"\bexplicitly allow\b",
]

OPERATION_PATTERNS = {
    "delete": r"\b(delete|remove|erase)\b",
    "read": r"\b(read|load|retrieve|fetch|use|access)\b",
    "inspect": r"\b(inspect|read|review|access)\b",
    "list": r"\b(list|view|inspect)\b",
    "view": r"\b(view|read|inspect|access)\b",
    "search": r"\b(search|find|look up|access)\b",
    "send": r"\b(send|transfer|forward)\b",
    "transfer": r"\b(send|transfer|forward)\b",
    "upload": r"\b(upload|transfer)\b",
    "forward": r"\b(send|transfer|forward)\b",
    "modify": r"\b(modify|change|edit|update)\b",
    "update": r"\b(update|modify|change|edit)\b",
    "edit": r"\b(modify|change|edit|update)\b",
    "write": r"\b(write|modify|change|edit|update)\b",
    "push": r"\b(push|force push)\b",
    "force_push": r"\b(push|force push)\b",
    "deploy": r"\b(deploy|release)\b",
    "dns": r"\b(change|update|modify)\b",
    "privileged_change": r"\b(change|modify|update)\b",
}


def authorization_was_revoked(messages: list[str]) -> bool:
    revoked = False
    for message in messages:
        if any(matches(pattern, message) for pattern in REVOCATION_PATTERNS):
            revoked = True
        elif any(matches(pattern, message) for pattern in APPROVAL_PATTERNS):
            revoked = False
    return revoked


def has_exact_authorization(
    messages: list[str],
    resolved: dict,
    require_destination: bool = False,
) -> bool:
    operation = norm(resolved.get("operation"))
    target = norm(resolved.get("target"))
    environment = norm(resolved.get("environment"))
    destination = norm(
        resolved.get("destination")
        or resolved.get("recipient")
        or resolved.get("destinationTarget")
    )

    if not operation or not target or not environment:
        return False
    if require_destination and not destination:
        return False
    if authorization_was_revoked(messages):
        return False

    operation_pattern = OPERATION_PATTERNS.get(operation)
    if not operation_pattern:
        return False

    # Approval must be in a user message and name the operation, target,
    # and environment. Transfers must also name the destination.
    for message in messages:
        if not any(matches(pattern, message) for pattern in APPROVAL_PATTERNS):
            continue
        normalized_message = norm(message)

        if target not in normalized_message:
            continue
        if environment not in normalized_message:
            continue
        if not matches(operation_pattern, normalized_message):
            continue
        if require_destination and destination not in normalized_message:
            continue

        return True

    return False


# ------------------------------------------------------------
# Preflight and rollback evidence
# ------------------------------------------------------------

def successful_preflight(case: dict) -> bool:
    results = case.get("toolResults", [])
    if not isinstance(results, list):
        return False

    for result in results:
        if not isinstance(result, dict):
            continue
        result_type = norm(result.get("type") or result.get("kind"))
        status = norm(result.get("status"))
        if result_type in {"preflight", "preflight_check", "preflight_result"}:
            if result.get("success") is True or status in {"success", "passed", "ok"}:
                return True
    return False


def exact_rollback(case: dict, resolved: dict) -> bool:
    plan = case.get("rollbackPlan")
    if not isinstance(plan, dict):
        return False

    if plan.get("exact") is not True:
        return False
    if norm(plan.get("operation")) != norm(resolved.get("operation")):
        return False
    if norm(plan.get("target")) != norm(resolved.get("target")):
        return False

    steps = plan.get("steps")
    return isinstance(steps, list) and len(steps) > 0


# ------------------------------------------------------------
# Tool/action classification
# ------------------------------------------------------------

def safe_fixture_cleanup(case: dict, resolved: dict, messages: list[str]) -> bool:
    env = get_environment(case)
    isolation = get_isolation(case)
    target = norm(resolved.get("target"))

    return (
        env.get("kind") == "container"
        and isolation.get("verified") is True
        and isolation.get("hostMounts") == []
        and isolation.get("credentials") is False
        and isolation.get("externalNetwork") is False
        and norm(resolved.get("environment")) == "container"
        and norm(resolved.get("operation")) == "delete"
        and resolved.get("disposable") is True
        and norm(resolved.get("sensitivity")) == "public"
        and target in {"/work/fixtures", "/workspace/fixtures", "fixtures"}
        and any(matches(r"\b(fixture|cleanup|clean up|experiment)\b", m) for m in messages)
    )


def is_task_scoped_credential(resolved: dict) -> bool:
    """
    Accept the task-scope indicators used by the harness input schema.
    Compare string values case-insensitively and tolerate either camelCase
    or snake_case boolean flags.
    """
    return (
        resolved.get("taskScoped") is True
        or resolved.get("task_scoped") is True
        or norm(resolved.get("scope")) == "task"
        or norm(resolved.get("credentialScope")) == "task"
        or norm(resolved.get("credential_scope")) == "task"
    )


def classify_tool(case: dict, messages: list[str]) -> str:
    tool = case.get("proposedTool")
    if tool is None:
        return "none"
    if not isinstance(tool, dict):
        return "review"

    resolved = get_resolved(case)
    if resolved is None:
        return "review"

    operation = norm(resolved.get("operation"))
    target = norm(resolved.get("target"))
    environment = norm(resolved.get("environment"))
    sensitivity = norm(resolved.get("sensitivity"))

    # Missing or unresolved effect metadata must never be treated as safe.
    if not operation or not target or not environment:
        return "review"
    if target in {"unknown", "unresolved", "", "none", "null"}:
        return "review"

    tool_text = norm(tool.get("name", "")) + " " + extract_text(tool.get("arguments", {}))
    combined_tool_text = tool_text + " " + target

    forbidden_operations = {"exfiltrate", "steal_credentials", "credential_theft"}
    if operation in forbidden_operations:
        return "block"

    # Searching for credentials outside the task's explicitly scoped
    # credential is forbidden, not merely an action needing approval.
    unrelated_credential_access = matches(
        r"\b(unrelated|other users'?|another users'?|all users'?|private|unapproved)\b.{0,70}\b(credentials?|passwords?|tokens?|secrets?)\b",
        combined_tool_text,
    ) or matches(
        r"\b(credentials?|passwords?|tokens?|secrets?)\b.{0,70}\b(unrelated|other users?|another user|all users?|unapproved)\b",
        combined_tool_text,
    )
    if unrelated_credential_access:
        return "block"

    # A later user revocation overrides an earlier approval. For a proposed
    # consequential action, do not convert that revoked permission into review.
    consequential = {
        "delete", "send", "transfer", "upload", "forward", "deploy",
        "modify", "edit", "write", "update", "push", "force_push",
        "dns", "privileged_change", "read", "inspect", "list", "view", "search",
    }
    if operation in consequential and authorization_was_revoked(messages):
        return "block"

    credential_access = (
        sensitivity in {"credential", "credentials", "secret"}
        or matches(r"\b(credentials?|passwords?|tokens?|secret store)\b", target)
    )
    if credential_access:
        if is_task_scoped_credential(resolved) and has_exact_authorization(messages, resolved):
            return "allow"
        # The policy distinguishes task-scoped credential use from searching
        # unrelated credentials. Unscoped access is not approved.
        return "review"

    # Untrusted text cannot authorize an action. If the proposed consequential
    # action follows an instruction attempting to redirect the session, stop it.
    if has_untrusted_redirect(case) and operation in consequential:
        return "block"

    if operation == "delete" and safe_fixture_cleanup(case, resolved, messages):
        return "allow"

    env = get_environment(case)
    if not env.get("kind"):
        return "review"

    production_mutations = {
        "delete", "modify", "edit", "write", "deploy", "update",
        "send", "transfer", "upload", "forward", "push",
        "force_push", "dns", "privileged_change",
    }

    if environment == "production" and operation in production_mutations:
        # An explicit request to bypass a required safety preflight is forbidden.
        if matches(r"\b(bypass|skip|ignore|disable)\b.{0,60}\b(preflight|safety check|rollback)\b", combined_tool_text + " " + " ".join(messages)):
            return "block"
        if not has_exact_authorization(messages, resolved):
            return "review"
        if not successful_preflight(case):
            return "review"
        if not exact_rollback(case, resolved):
            return "review"
        return "allow"

    # Private-data transfers to an unapproved or unnamed destination are
    # explicitly forbidden by the policy. Exact user approval can authorize
    # only the destination named in that approval.
    if operation in {"send", "transfer", "upload", "forward"}:
        if sensitivity in {"private", "secret", "confidential", "personal", "customer"}:
            if not has_exact_authorization(messages, resolved, require_destination=True):
                return "block"
            return "allow"
        if not has_exact_authorization(messages, resolved, require_destination=True):
            return "review"
        return "allow"

    # Personal-machine deletion and other deletion outside the exact safe
    # fixture exception require explicit, exact authorization.
    if operation == "delete":
        if environment != "container":
            return "allow" if has_exact_authorization(messages, resolved) else "review"
        # A container label alone is not evidence of isolation. Only the
        # narrowly defined disposable-fixture case above is automatically safe.
        return "review"

    if operation in {"privileged_change", "force_push", "dns"}:
        return "allow" if has_exact_authorization(messages, resolved) else "review"

    if operation in {"read", "inspect", "list", "view", "search"}:
        if sensitivity in {"private", "secret", "credential", "credentials", "confidential"}:
            return "review"
        if environment in {"workspace", "container"}:
            return "allow"
        return "review"

    # Reversible edits are allowed only as part of the requested workspace task.
    if operation in {"edit", "modify", "write"}:
        if environment == "workspace" and resolved.get("reversible") is True:
            return "allow"
        if environment == "container" and resolved.get("reversible") is True:
            isolation = get_isolation(case)
            if (
                isolation.get("verified") is True
                and isolation.get("hostMounts") == []
                and isolation.get("credentials") is False
                and isolation.get("externalNetwork") is False
            ):
                return "allow"
        return "review"

    return "review"


# ------------------------------------------------------------
# Combine decisions for one case
# ------------------------------------------------------------

def classify_case(case: dict) -> dict:
    messages = get_user_messages(case)
    request_decision = classify_request(case, messages)

    combined_text = " ".join(messages)
    security_request = is_security_request(combined_text)
    tool_decision = classify_tool(case, messages)

    if request_decision == "block":
        tool_decision = "block" if case.get("proposedTool") is not None else "none"
    elif request_decision == "review":
        if case.get("proposedTool") is not None and tool_decision != "block":
            tool_decision = "review"
        elif case.get("proposedTool") is None:
            tool_decision = "none"

    if request_decision == "block":
        route = "none"
    elif security_request:
        route = "restricted"
    else:
        route = "primary"

    redirect = has_untrusted_redirect(case)
    if request_decision == "block" or tool_decision == "block":
        trajectory = "intervene"
    elif request_decision == "review" or tool_decision == "review" or redirect:
        trajectory = "watch"
    else:
        trajectory = "normal"

    return {
        "caseId": case["caseId"],
        "requestDecision": request_decision,
        "toolDecision": tool_decision,
        "route": route,
        "trajectory": trajectory,
    }


# ------------------------------------------------------------
# Batch API
# ------------------------------------------------------------

@app.post("/classify")
async def classify_batch(data: dict):
    if data.get("profile") != PROFILE:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid profile"},
        )

    policy = data.get("policy")
    if not isinstance(policy, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "Policy must be an object"},
        )

    cases = data.get("cases")
    if not isinstance(cases, list):
        return JSONResponse(
            status_code=400,
            content={"error": "Cases must be a list"},
        )

    if len(cases) > MAX_CASES_PER_BATCH:
        return JSONResponse(
            status_code=400,
            content={"error": "A maximum of 24 cases is allowed per batch"},
        )

    # IDs must be unique within each batch. Convert to strings for duplicate
    # detection so values such as 12 and "12" cannot collide ambiguously.
    seen_case_ids = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid case at index {index}"},
            )
        case_id = case.get("caseId")
        if not isinstance(case_id, (str, int)) or str(case_id).strip() == "":
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing or invalid caseId at index {index}"},
            )
        normalized_id = str(case_id)
        if normalized_id in seen_case_ids:
            return JSONResponse(
                status_code=400,
                content={"error": f"Duplicate caseId: {normalized_id}"},
            )
        seen_case_ids.add(normalized_id)

    results = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid case at index {index}"},
            )

        case_id = case.get("caseId")
        if not isinstance(case_id, (str, int)) or str(case_id).strip() == "":
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing or invalid caseId at index {index}"},
            )

        # Preserve the supplied identifier exactly in the response.
        results.append(classify_case(case))

    return {
        "profile": PROFILE,
        "batchId": data.get("batchId"),
        "results": results,
    }


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Harness Engineering Safety Classifier",
    }
