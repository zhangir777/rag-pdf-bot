from bot.main import before_breadcrumb, before_send


def test_before_send_drops_aiogram_fetch_updates():
    event = {
        "logger": "aiogram.dispatcher",
        "logentry": {
            "formatted": "Failed to fetch updates - TelegramNetworkError: HTTP Client says - Request timeout error"
        },
    }
    assert before_send(event, {}) is None

    event_msg = {
        "logger": "aiogram.dispatcher",
        "logentry": {"message": "Failed to fetch updates - TelegramNetworkError"},
    }
    assert before_send(event_msg, {}) is None


def test_before_send_keeps_other_events():
    other_logger = {
        "logger": "root",
        "logentry": {"formatted": "Failed to fetch updates"},
    }
    assert before_send(other_logger, {}) is other_logger

    other_msg = {
        "logger": "aiogram.dispatcher",
        "logentry": {"formatted": "Unhandled exception"},
    }
    assert before_send(other_msg, {}) is other_msg
    assert before_send({}, {}) == {}


def test_before_breadcrumb_scrubs_bot_token():
    crumb = {
        "message": "POST https://api.telegram.org/bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11/getUpdates",
        "data": {
            "url": "https://api.telegram.org/bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11/getUpdates",
            "status_code": 200,
        },
    }
    res = before_breadcrumb(crumb, {})
    assert res["message"] == "POST https://api.telegram.org/bot<hidden>/getUpdates"
    assert res["data"]["url"] == "https://api.telegram.org/bot<hidden>/getUpdates"
    assert res["data"]["status_code"] == 200
