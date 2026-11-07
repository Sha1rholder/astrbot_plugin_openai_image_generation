"""Test the Image API contract and persistence."""

import base64
import io
import json
from pathlib import Path

import httpx
import jsonschema
import pytest
import yaml
from data.plugins.astrbot_plugin_openai_image_generation.src.generate import (
    REQUEST_SCHEMA,
    ImageGenerationError,
)
from data.plugins.astrbot_plugin_openai_image_generation.src.image_api.generate import (
    INPUT_SCHEMA,
    ImageAPI,
)
from PIL import Image

ARGUMENTS = {"description": "A cat", "prompt": {"text": "cat"}}


def test_defaults_and_fixed_fields(api):
    payload, refs = api.prepare(
        {"description": "A cat", "prompt": {"text": " Draw a cat "}}
    )
    assert refs == []
    assert payload == {
        "model": "gpt-image-2.6-sunburst",
        "prompt": "Draw a cat",
        "size": "auto",
        "quality": "auto",
        "complexity": "high",
        "background": "auto",
        "n": 1,
        "output_format": "webp",
        "output_compression": 100,
        "moderation": "low",
        "partial_images": 0,
        "stream": False,
    }


@pytest.mark.parametrize("size", ["1024x640", "3840x2160", "768x2304"])
def test_size_boundaries(api, size):
    payload, _ = api.prepare({**ARGUMENTS, "size": size})
    assert payload["size"] == size


@pytest.mark.parametrize(
    "options",
    [
        {"prompt": "cat"},
        {"model": "gpt-image-1"},
        {"number_of_images": 0},
        {"number_of_images": 6},
        {"prompt": {"text": "cat", "images": [1, 1]}},
        {"prompt": {"text": "cat", "images": ["/etc/passwd"]}},
        {"stream": True},
        {"api": "chat"},
        {"llm": "gpt-6-astra"},
        {"previous_response_id": "resp_previous"},
        {"reasoning": {"effort": "high"}},
        {"reasoning_effort": "high"},
        {"moderation": "auto"},
        {"output_format": "png"},
        {"output_compression": 80},
    ],
)
def test_schema_rejects_unsupported_arguments(api, options):
    with pytest.raises(ImageGenerationError, match="Invalid image arguments"):
        api.prepare({**ARGUMENTS, **options})


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"size": "1920x1080"}, "multiples of 16"),
        ({"size": "4096x2048"}, "between 16 and 3840"),
        ({"size": "2048x640"}, "aspect ratio"),
        ({"size": "512x512"}, "pixel count"),
        ({"size": "3840x3840"}, "pixel count"),
        ({"description": "   "}, "description must not be blank"),
        ({"prompt": {"text": "   "}}, "prompt must not be blank"),
    ],
)
def test_invalid_values(api, options, reason):
    with pytest.raises(ImageGenerationError, match=reason):
        api.prepare({**ARGUMENTS, **options})


def test_null_options_use_defaults(api):
    default, _ = api.prepare(ARGUMENTS)
    options = dict.fromkeys(set(INPUT_SCHEMA["properties"]) - {"prompt", "description"})
    actual, _ = api.prepare({**ARGUMENTS, **options})
    assert actual == default


@pytest.mark.asyncio
@pytest.mark.parametrize("with_reference", [False, True])
async def test_api_contract_and_original_bytes(api, tmp_path, webp, with_reference):
    encoded = base64.b64encode(webp).decode()
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": encoded}] * 2})

    api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/",
        transport=httpx.MockTransport(respond),
    )
    arguments = {
        "description": "A cat",
        "prompt": {"text": "  cat  ", "images": [1] if with_reference else []},
        "model": "flare",
        "quality": "max",
        "complexity": "low",
        "size": None,
        "background_transparency": "transparent",
        "number_of_images": 2,
    }
    (api.plugin_dir / "metadata.yaml").write_text("version: v9.8.7\n")
    payload, _ = api.prepare(arguments)
    paths = await api.generate(
        payload,
        [encoded] if with_reference else [],
        tmp_path,
        arguments,
    )
    archive_text = (paths[0].parent / "request.yaml").read_text()
    assert archive_text.splitlines()[0] == (
        "# yaml-language-server: $schema=" + REQUEST_SCHEMA["$id"]
    )
    archive = yaml.safe_load(archive_text)
    jsonschema.validate(archive, REQUEST_SCHEMA)
    assert archive == {
        "plugin_version": "v9.8.7",
        "api": "image",
        "request": arguments,
    }
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == (
        "/v1/images/edits" if with_reference else "/v1/images/generations"
    )
    body = json.loads(request.content)
    expected = dict(payload)
    if with_reference:
        expected["images"] = [{"image_url": "data:image/webp;base64," + encoded}]
    assert body == expected
    assert body["model"] == "gpt-image-2.6-flare"
    assert body["quality"] == "max"
    assert body["complexity"] == "low"
    assert body["background"] == "transparent"
    assert len(set(paths)) == 2
    assert all(path.read_bytes() == webp and path.suffix == ".webp" for path in paths)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 429, 500])
async def test_errors_do_not_leak_response_or_retry(api, tmp_path, status):
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(
            status,
            text="secret upstream content",
            headers={"location": "https://elsewhere.invalid"},
        )

    api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/",
        transport=httpx.MockTransport(respond),
    )
    payload, _ = api.prepare(ARGUMENTS)
    with pytest.raises(ImageGenerationError) as error:
        await api.generate(payload, [], tmp_path, ARGUMENTS)
    assert str(status) in str(error.value)
    assert "secret" not in str(error.value)
    assert len(attempts) == 1
    directory = tmp_path / "0"
    assert {path.name for path in directory.iterdir()} == {"request.yaml"}


@pytest.mark.parametrize(
    "result", [{"data": []}, {"data": [{"url": "https://example.invalid/x"}]}]
)
def test_invalid_responses_preserve_request(api, tmp_path, result):
    directory = api.create_request(tmp_path / "out", ARGUMENTS)
    with pytest.raises(ImageGenerationError):
        ImageAPI.save_images(result, directory, 1)
    assert {path.name for path in directory.iterdir()} == {"request.yaml"}


def test_invalid_second_image_preserves_archive_without_partial_output(
    api, tmp_path, webp
):
    directory = api.create_request(tmp_path / "out", ARGUMENTS)
    with pytest.raises(ImageGenerationError, match="valid WebP"):
        ImageAPI.save_images(
            {
                "data": [
                    {"b64_json": base64.b64encode(webp).decode()},
                    {"b64_json": "invalid-base64!"},
                ]
            },
            directory,
            2,
        )
    assert {path.name for path in directory.iterdir()} == {"request.yaml"}


def test_non_webp_output_is_rejected(tmp_path):
    output = io.BytesIO()
    Image.new("RGB", (16, 16)).save(output, "PNG")
    encoded = base64.b64encode(output.getvalue()).decode()
    with pytest.raises(ImageGenerationError):
        ImageAPI.save_images(
            {"data": [{"b64_json": encoded}]},
            tmp_path / "out",
            1,
        )
    assert ImageAPI.reference_payload([encoded])[0]["image_url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["config", "environment"])
async def test_key_sources(api, tmp_path, monkeypatch, webp, source):
    key = "test-key-not-a-real-secret"
    if source == "config":
        api.config["api_key"] = " " + key + "\n"
        monkeypatch.setenv("OPENAI_API_KEY", "lower-priority")
    else:
        api.config["api_key"] = " \t"
        monkeypatch.setenv("OPENAI_API_KEY", " " + key + "\n")
    captured = []
    client_type = httpx.AsyncClient

    def respond(request):
        captured.append(request.headers["authorization"])
        return httpx.Response(
            200, json={"data": [{"b64_json": base64.b64encode(webp).decode()}]}
        )

    def create_client(**kwargs):
        return client_type(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "AsyncClient", create_client)
    payload, _ = api.prepare(ARGUMENTS)
    await api.generate(payload, [], tmp_path / "out", ARGUMENTS)
    assert captured == ["Bearer " + key]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,value",
    [
        ("base_url", "ftp://example.invalid"),
        ("base_url", "https://user:secret@example.invalid"),
        ("timeout_seconds", 0),
        ("timeout_seconds", float("nan")),
    ],
)
async def test_invalid_client_config_prevents_request(api, tmp_path, key, value):
    api.config.update(api_key="test-key", **{key: value})
    payload, _ = api.prepare(ARGUMENTS)
    with pytest.raises(ImageGenerationError, match=key) as error:
        await api.generate(payload, [], tmp_path, ARGUMENTS)
    assert "secret" not in str(error.value)
    assert api.client is None


@pytest.mark.parametrize("value", [None, "", " \t"])
def test_empty_proxy_preserves_environment_settings(api, value):
    api.config["proxy"] = value
    assert api.proxy_url() is None


@pytest.mark.parametrize(
    "value",
    [
        "ftp://user:private-password@localhost:1080",
        "localhost:1080",
        "socks5h://",
        "http://localhost:invalid",
        "http://localhost:65536",
        "http://localhost:0",
        "http://localhost/path",
        "http://localhost?password=private-password",
        "http://localhost#private-password",
        123,
    ],
)
def test_invalid_proxy_does_not_leak_credentials(api, value):
    api.config["proxy"] = value
    with pytest.raises(ImageGenerationError, match="proxy") as error:
        api.proxy_url()
    assert "private-password" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["http", "socks5", "socks5h"])
async def test_requests_use_authenticated_proxy(api, tmp_path, monkeypatch, scheme):
    """Pass explicit proxy credentials to the HTTP client without relying on the environment."""
    proxy = f"{scheme}://user:pass%40word@localhost:1080"
    api.config["proxy"] = " " + proxy + " "
    monkeypatch.setenv("HTTP_PROXY", "http://localhost:1")
    monkeypatch.setenv("NO_PROXY", "*")
    client_type = httpx.AsyncClient
    options = []

    def create_client(**kwargs):
        options.append(kwargs)
        return client_type(
            base_url=kwargs["base_url"],
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )

    monkeypatch.setattr(httpx, "AsyncClient", create_client)
    payload, _ = api.prepare(ARGUMENTS)
    with pytest.raises(ImageGenerationError, match="HTTP 500"):
        await api.generate(payload, [], tmp_path / "out", ARGUMENTS)
    assert len(options) == 1
    assert options[0]["proxy"] == httpx.URL(proxy)
    assert options[0]["follow_redirects"] is False


@pytest.mark.parametrize("limit", [0, True, "10"])
def test_invalid_storage_limit(api, limit):
    api.config["max_storage_gb"] = limit
    with pytest.raises(ImageGenerationError, match="max_storage_gb"):
        api.storage_limit()


def test_storage_failure_keeps_previous_request(api, tmp_path, webp, monkeypatch):
    arguments = ARGUMENTS
    result = {"data": [{"b64_json": base64.b64encode(webp).decode()}]}
    output = tmp_path / "images"
    directory = api.create_request(output, arguments)
    ImageAPI.save_images(result, directory, 1)
    original = Path.write_text

    def fail_metadata(path, *args, **kwargs):
        if path.name == "request.yaml":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(OSError):
        api.create_request(output, arguments)
    assert {p.name for p in output.iterdir()} == {"0"}
    assert (output / "0" / "0.webp").read_bytes() == webp


def test_image_storage_failure_preserves_request(api, tmp_path, webp, monkeypatch):
    arguments = ARGUMENTS
    directory = api.create_request(tmp_path / "images", arguments)
    archive = (directory / "request.yaml").read_bytes()
    original = Path.replace

    def fail_second_image(path, *args, **kwargs):
        if path.name == "1.webp":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", fail_second_image)
    result = {"data": [{"b64_json": base64.b64encode(webp).decode()}] * 2}
    with pytest.raises(OSError):
        ImageAPI.save_images(result, directory, 2)
    assert {path.name for path in directory.iterdir()} == {"request.yaml"}
    assert (directory / "request.yaml").read_bytes() == archive


def test_cleanup_ignores_unrelated_files_and_symlinks(tmp_path):
    output = tmp_path / "images"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    (output / "0").symlink_to(outside, target_is_directory=True)
    for name in ["1", "2", "02", "old-uuid"]:
        (output / name).mkdir()
    (output / "old.webp").write_text("keep")
    (output / "1" / "large").write_bytes(b"x" * 100)
    ImageAPI.prune_requests(output, 10, 2)
    assert not (output / "1").exists()
    assert (outside / "keep").read_text() == "keep"
    assert {p.name for p in output.iterdir()} == {
        "0",
        "2",
        "02",
        "old-uuid",
        "old.webp",
    }


def test_next_request_id_scans_only_numeric_directories(tmp_path):
    assert ImageAPI.next_request_id(tmp_path / "missing") == 0
    for name in ["2", "9", "10", "notes"]:
        (tmp_path / name).mkdir()
    (tmp_path / "999").write_text("not a directory")
    (tmp_path / "1000").symlink_to(tmp_path / "2", target_is_directory=True)
    assert ImageAPI.next_request_id(tmp_path) == 11


@pytest.mark.parametrize(
    "limit,remaining", [(100, {"0", "2", "10"}), (99, {"2", "10"}), (60, {"10"})]
)
def test_capacity_cleanup_order_and_boundary(tmp_path, limit, remaining):
    for name, size in [("10", 30), ("2", 30), ("0", 40)]:
        directory = tmp_path / name / "nested"
        directory.mkdir(parents=True)
        (directory / "image.webp").write_bytes(b"x" * size)
    ImageAPI.prune_requests(tmp_path, limit, 11)
    assert {p.name for p in tmp_path.iterdir()} == remaining


@pytest.mark.asyncio
async def test_cleanup_precedes_http_and_keeps_current_and_queued(
    api, tmp_path, monkeypatch
):
    arguments = ARGUMENTS
    output = tmp_path / "images"
    old = api.create_request(output, arguments)
    (old / "large.webp").write_bytes(b"x" * 10000)
    current = api.create_request(output, arguments)
    queued = api.create_request(output, arguments)
    monkeypatch.setattr(
        api, "storage_limit", lambda: ImageAPI.storage_size(current) * 2 + 1
    )
    calls = []

    def respond(request):
        assert not old.exists()
        assert (current / "request.yaml").exists()
        assert (queued / "request.yaml").exists()
        calls.append(request)
        return httpx.Response(500)

    api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/",
        transport=httpx.MockTransport(respond),
    )
    payload, _ = api.prepare(arguments)
    with pytest.raises(ImageGenerationError, match="HTTP 500"):
        await api.generate(payload, [], output, arguments, current)
    assert len(calls) == 1
    monkeypatch.setattr(api, "storage_limit", lambda: 1)
    with pytest.raises(ImageGenerationError, match="No API request"):
        await api.generate(payload, [], output, arguments, current)
    assert len(calls) == 1
    assert current.exists() and queued.exists()


@pytest.mark.parametrize(
    "config,expected",
    [({}, 10_000_000_000), ({"max_storage_gb": 25}, 25_000_000_000)],
)
def test_storage_limit_in_bytes(api, config, expected):
    api.config.update(config)
    assert api.storage_limit() == expected
