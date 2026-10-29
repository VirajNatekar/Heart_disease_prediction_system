"""Explainable rule-based analysis for a defensive fraud-awareness demo."""

from __future__ import annotations

import re
from urllib.parse import urlparse


URL_PATTERN = re.compile(r"""(?:https?://|www\.)[^\s<>"']+""", re.IGNORECASE)
URGENT_TERMS = (
    "immediately",
    "urgent",
    "right now",
    "today",
    "will be blocked",
    "will be suspended",
    "expire",
)
SENSITIVE_TERMS = (
    "password",
    "one-time password",
    "otp",
    "pin",
    "cvv",
    "verify your account",
    "kyc",
    "login details",
)
PAYMENT_TERMS = (
    "pay now",
    "payment",
    "transfer money",
    "send money",
    "fee",
    "gift card",
)
SHORTENER_HOSTS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd"}


def _extract_urls(text: str) -> list[str]:
    return [match.rstrip(".,!?;:)]}") for match in URL_PATTERN.findall(text)]


def _host_for_url(url: str) -> str | None:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = (parsed.hostname or "").rstrip(".").lower()
    return host or None


def analyze_message(text: str, previously_seen_hosts: set[str] | None = None) -> dict:
    """Return a heuristic score, evidence, extracted hosts, and a cautious recommendation."""
    normalized = text.casefold()
    evidence: list[dict[str, str | int]] = []
    score = 0

    def add_signal(label: str, explanation: str, points: int) -> None:
        nonlocal score
        score += points
        evidence.append({"label": label, "explanation": explanation, "points": points})

    if any(term in normalized for term in URGENT_TERMS):
        add_signal("Urgent language", "The message pressures the recipient to act quickly.", 2)

    if any(term in normalized for term in SENSITIVE_TERMS):
        add_signal(
            "Sensitive information",
            "The message mentions credentials, account verification, or security codes.",
            3,
        )

    if any(term in normalized for term in PAYMENT_TERMS):
        add_signal("Payment request", "The message mentions a payment or money transfer.", 2)

    urls = _extract_urls(text)
    hosts = sorted({host for url in urls if (host := _host_for_url(url))})
    if urls:
        add_signal("Link included", "The message contains a link. Inspect it carefully before opening.", 1)

    for url in urls:
        host = _host_for_url(url)
        if not host:
            continue
        parsed = urlparse(url if "://" in url else f"http://{url}")
        if parsed.scheme.casefold() == "http":
            add_signal(
                "Unencrypted link",
                f"The link to {host} uses HTTP rather than HTTPS.",
                1,
            )
        if host in SHORTENER_HOSTS:
            add_signal(
                "Shortened link",
                f"The destination of the shortened link to {host} is not visible here.",
                2,
            )
        if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
            add_signal(
                "IP-address link",
                "The link uses a numeric IP address instead of a readable domain name.",
                2,
            )

    seen_hosts = previously_seen_hosts or set()
    repeated = sorted(set(hosts).intersection(seen_hosts))
    if repeated:
        add_signal(
            "Previously reported host",
            f"This report includes a host seen in an earlier local report: {', '.join(repeated)}.",
            2,
        )

    if score >= 5:
        level = "High"
        recommendation = "Do not open the link or share information. Verify the request using the organization's official app or contact details."
    elif score >= 2:
        level = "Medium"
        recommendation = "Pause and verify the sender and request through a separate, official channel before acting."
    else:
        level = "Low"
        recommendation = "No strong warning signs were detected by this demo. That does not prove the message or link is safe."

    return {
        "score": score,
        "level": level,
        "evidence": evidence,
        "hosts": hosts,
        "recommendation": recommendation,
    }
