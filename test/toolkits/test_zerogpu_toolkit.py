# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

import pytest

from camel.toolkits import ZeroGPUToolkit


@pytest.fixture
def toolkit():
    return ZeroGPUToolkit(
        api_key="zgpu-api-test",
        project_id="test-project",
        base_url="https://api.zerogpu.ai",
    )


def test_get_tools(toolkit):
    tools = toolkit.get_tools()

    assert len(tools) == 11

    tool_names = {tool.func.__name__ for tool in tools}

    expected = {
        "zerogpu_chat",
        "zerogpu_chat_thinking",
        "zerogpu_summarize",
        "zerogpu_classify_iab",
        "zerogpu_classify_iab_enriched",
        "zerogpu_classify_zero_shot",
        "zerogpu_classify_structured",
        "zerogpu_extract_entities",
        "zerogpu_extract_json",
        "zerogpu_extract_pii",
        "zerogpu_redact_pii",
    }

    assert tool_names == expected


def test_invalid_api_key():
    with pytest.raises(ValueError):
        ZeroGPUToolkit(api_key="bad", project_id="test")


def test_chat(toolkit, monkeypatch):
    captured = {}

    def mock_create_response(**kwargs):
        captured.update(kwargs)
        return {"output": [{"content": [{"text": "hello world"}]}]}

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    result = toolkit.zerogpu_chat("hello")

    assert result == "hello world"
    assert captured["model"] == "LFM2.5-1.2B-Instruct"


def test_chat_with_system(toolkit, monkeypatch):
    captured = {}

    def mock_create_response(**kwargs):
        captured.update(kwargs)
        return {"output": [{"content": [{"text": "ok"}]}]}

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    toolkit.zerogpu_chat("hello", system="You are helpful")

    assert isinstance(captured["input"], list)
    assert captured["input"][0]["role"] == "system"
    assert captured["input"][1]["role"] == "user"


def test_summarize(toolkit, monkeypatch):
    captured = {}

    def mock_create_response(**kwargs):
        captured.update(kwargs)
        return {"output": [{"content": [{"text": "summary"}]}]}

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    result = toolkit.zerogpu_summarize("text")

    assert result == "summary"
    assert captured["model"] == "llama-3.1-8b-instruct-fast"


def test_classify_iab(toolkit, monkeypatch):
    captured = {}

    def mock_create_response(**kwargs):
        captured.update(kwargs)
        return {"output": [{"content": [{"text": '{"category": "sports"}'}]}]}

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    result = toolkit.zerogpu_classify_iab("text")

    assert result == {"category": "sports"}
    assert captured["model"] == "zlm-v1-iab-classify-edge"
    assert captured["input"] == "text"


def test_classify_zero_shot(toolkit, monkeypatch):
    captured = {}

    def mock_create_response(**kwargs):
        captured.update(kwargs)
        return {"output": [{"content": [{"text": '{"sports": 0.9}'}]}]}

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    toolkit.zerogpu_classify_zero_shot("text", ["sports"])

    assert captured["categories"] == ["sports"]


def test_extract_pii_categories(toolkit, monkeypatch):
    captured = {}

    def mock_create_response(**kwargs):
        captured.update(kwargs)
        return {"output": [{"content": [{"text": '{"entities": []}'}]}]}

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    toolkit.zerogpu_extract_pii("text", threshold=0.5, categories=["email"])

    metadata = captured["metadata"]

    assert metadata["categories"] == ["email"]
    assert metadata["threshold"] == 0.5


def test_extract_pii_no_threshold(toolkit, monkeypatch):
    captured = {}

    def mock_create_response(**kwargs):
        captured.update(kwargs)
        return {"output": [{"content": [{"text": '{"entities": []}'}]}]}

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    toolkit.zerogpu_extract_pii("text")

    metadata = captured["metadata"]

    assert "threshold" not in metadata


def test_extract_pii_no_categories(toolkit, monkeypatch):
    captured = {}

    def mock_create_response(**kwargs):
        captured.update(kwargs)
        return {"output": [{"content": [{"text": '{"entities": []}'}]}]}

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    toolkit.zerogpu_extract_pii("text")

    metadata = captured["metadata"]

    assert "categories" not in metadata


def test_redact_pii(toolkit, monkeypatch):
    captured = {}

    def mock_create_response(**kwargs):
        captured.update(kwargs)
        return {"output": [{"content": [{"text": "Hello [PERSON]"}]}]}

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    result = toolkit.zerogpu_redact_pii("text")

    metadata = captured["metadata"]

    assert metadata["usecase"] == "redact"
    assert metadata["mask"] == "label"
    assert result == "Hello [PERSON]"


def test_error_mapping(toolkit, monkeypatch):
    def mock_create_response(**kwargs):
        raise Exception("auth error")

    monkeypatch.setattr(
        toolkit.client.responses,
        "create_response",
        mock_create_response,
    )

    # dict-returning
    result_dict = toolkit.zerogpu_classify_iab("text")
    assert "error" in result_dict

    # str-returning
    result_str = toolkit.zerogpu_chat("text")
    assert "Error" in result_str
