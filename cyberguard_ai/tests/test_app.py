import pytest

from app import app


@pytest.fixture
def client(tmp_path):
    app.config.update(
        TESTING=True,
        DATABASE=str(tmp_path / "test.db"),
    )
    with app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Analyze a message" in response.data


def test_post_analyzes_and_stores_message(client):
    response = client.post(
        "/",
        data={"message": "Your account is urgent, verify your password http://example.test"},
    )

    assert response.status_code == 200
    assert b"High risk" in response.data
    assert b"Sensitive information" in response.data


def test_repeat_host_is_correlated(client):
    client.post("/", data={"message": "Look at http://example.test"})
    response = client.post("/", data={"message": "Another message: http://example.test"})

    assert b"Medium risk" in response.data
    assert b"Previously reported host" in response.data


def test_blank_message_shows_validation_error(client):
    response = client.post("/", data={"message": "  "})

    assert response.status_code == 200
    assert b"Enter a message to analyze." in response.data


def test_clear_history_removes_reports(client):
    client.post("/", data={"message": "Look at http://example.test"})
    response = client.post("/clear-history", follow_redirects=True)

    assert response.status_code == 200
    assert b"Local report history cleared." in response.data
    assert b"No reports yet" in response.data


def test_evidence_json_is_valid(client):
    client.post("/", data={"message": "Urgent payment now http://example.test"})
    with app.app_context():
        from app import get_db

        row = get_db().execute("SELECT evidence FROM reports").fetchone()
    assert b"Payment request" in client.get("/").data
