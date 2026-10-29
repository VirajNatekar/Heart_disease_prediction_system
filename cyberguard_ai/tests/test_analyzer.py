from analyzer import analyze_message


def test_suspicious_message_explains_multiple_signals():
    result = analyze_message(
        "Your account will be blocked today. Verify your password and pay now: "
        "http://example.test/login"
    )

    assert result["level"] == "High"
    labels = {item["label"] for item in result["evidence"]}
    assert {"Urgent language", "Sensitive information", "Payment request", "Unencrypted link"} <= labels
    assert result["hosts"] == ["example.test"]


def test_repeated_host_increases_risk():
    result = analyze_message("Please review http://example.test", {"example.test"})

    assert result["level"] == "High"
    assert any(item["label"] == "Previously reported host" for item in result["evidence"])


def test_message_without_indicators_is_not_called_safe():
    result = analyze_message("Team lunch has moved to noon.")

    assert result["level"] == "Low"
    assert "does not prove" in result["recommendation"]


def test_malformed_url_does_not_crash():
    result = analyze_message("See https:// for details")

    assert result["level"] == "Low"
    assert result["hosts"] == []
