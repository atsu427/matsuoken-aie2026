from types import SimpleNamespace

from app import ask


def test_ask_displays_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="回答",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12, candidates_token_count=34, total_token_count=46
        ),
    )
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **kwargs: response)
        ),
    )
    timestamps = iter((10.0, 11.25))
    monkeypatch.setattr(ask.time, "perf_counter", lambda: next(timestamps))

    assert ask.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "[usage] API呼び出し試行: 1回" in output
    assert "入力: 12 / 出力: 34 / 合計: 46 トークン" in output
    assert "応答時間: 1.25 秒" in output


def test_ask_reports_missing_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(text="回答", usage_metadata=None)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **kwargs: response)
        ),
    )
    timestamps = iter((10.0, 10.5))
    monkeypatch.setattr(ask.time, "perf_counter", lambda: next(timestamps))

    assert ask.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "入力・出力・合計トークン数: 未取得" in output
    assert "応答時間: 0.50 秒" in output


def test_ask_reports_attempt_and_response_time_when_api_call_fails(monkeypatch, capsys):
    def fail_request(**kwargs):
        raise RuntimeError("API request failed")

    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=fail_request)
        ),
    )
    timestamps = iter((10.0, 11.5))
    monkeypatch.setattr(ask.time, "perf_counter", lambda: next(timestamps))

    try:
        ask.ask("質問")
    except RuntimeError as error:
        assert str(error) == "API request failed"
    else:
        raise AssertionError("Expected the API error to propagate")

    output = capsys.readouterr().out
    assert "[usage] API呼び出し試行: 1回" in output
    assert "入力・出力・合計トークン数: 未取得" in output
    assert "応答時間: 1.50 秒" in output
