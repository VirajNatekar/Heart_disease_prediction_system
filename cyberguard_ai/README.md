# CyberGuard AI — Mini Project

An explainable, local web prototype that reviews user-submitted messages and links for common fraud indicators. It assigns a transparent risk level, explains the evidence, and notes when a link host has appeared in an earlier report.

> This is an educational demonstration, not a production security product. It uses transparent rules rather than a trained AI model, makes no live reputation lookups, and cannot prove that a message is fraudulent or safe.

## Project abstract

Digital scams can combine urgent messages, requests for credentials or payments, and deceptive links. CyberGuard AI demonstrates a small investigation workflow for user-submitted content: it extracts links, checks for a few explainable warning signs, correlates repeated link hosts across local reports, and presents a risk assessment with practical safety advice. The prototype is designed to show how an AI-agent workflow can coordinate evidence collection and risk assessment without silently taking action on a user's device.

## Objectives

- Accept a suspicious message pasted by the user.
- Identify common social-engineering and link warning signs.
- Explain the evidence behind the risk score.
- Correlate a link host with earlier locally stored reports.
- Give a cautious recommendation rather than claiming certainty.

## Technology and modules

- **Python and Flask:** local web app and request handling.
- **Rule-based analyzer:** extracts URLs and scores message/link indicators.
- **SQLite:** stores submitted reports locally for repeated-host correlation.
- **HTML and CSS:** simple responsive user interface.
- **Tests:** unit tests for analysis and Flask route tests.

## How the prototype works

1. The user pastes a message into the form.
2. The analyzer looks for urgency, requests for sensitive information, payment pressure, and URLs.
3. It inspects URL features without opening or contacting the destination.
4. It checks whether any extracted host appeared in an earlier local report.
5. It saves the report and displays a score, risk level, evidence, and recommendation.

The score is an educational heuristic, not a probability. A higher score means more warning signs were detected; it does not establish that fraud occurred.

## Run on Windows

From the repository root, install the existing project dependencies if Flask is not already available:

```powershell
python -m pip install -r requirements.txt
python cyberguard_ai\app.py
```

Open `http://127.0.0.1:5000` in a browser. Reports are stored in `cyberguard_ai\instance\cyberguard.db`. The app binds to localhost by default.

Run the focused tests from the repository root:

```powershell
python -m pytest cyberguard_ai\tests -q
```

## Safe demo input

```text
Your account will be blocked today. Verify your password and pay the pending fee now: http://example.test/verify
```

Use only fictional or otherwise safe example text. The application does not visit the URL.

## Limitations and future work

- Rules can miss new scams and can flag legitimate messages.
- No SMS, email, phone, or other app is monitored automatically.
- No external threat-intelligence service or machine-learning model is connected.
- Reports are stored locally without authentication; use fictional data and do not expose this development server to the internet.
- Possible next steps for a larger supervised project include a vetted labeled dataset, an opt-in threat-intelligence integration, user-controlled report deletion, and evaluation of precision/recall. Keep user data minimized and obtain consent before collection.
