import base64
import io
import json

import httpx
import pytest
from conftest import api_module
from PIL import Image

ImageAPI = api_module.ImageAPI
ImageGenerationError = api_module.ImageGenerationError


@pytest.fixture
def webp():
    """Return lossless WebP bytes for a small image with transparency."""
    output = io.BytesIO()
    Image.new("RGBA", (16, 16), (20, 40, 60, 128)).save(output, "WEBP", lossless=True)
    return output.getvalue()


def test_defaults_and_fixed_fields():
    """Verify prompt trimming and the default and fixed API request fields."""
    payload, refs = ImageAPI.prepare({"prompt": {"text": " Draw a cat "}})
    assert refs == []
    assert payload == {
        "model": "gpt-image-2.6-flare",
        "prompt": "Draw a cat",
        "size": "auto",
        "quality": "medium",
        "complexity": "high",
        "background": "opaque",
        "n": 1,
        "output_format": "webp",
        "output_compression": 100,
        "moderation": "low",
        "partial_images": 0,
        "stream": False,
    }


@pytest.mark.parametrize(
    "size",
    [
        "1024x640",
        "3840x2160",
        "2160x3840",
        "768x2304",
        "1920x1088",
        "1024x1024",
    ],
)
def test_size_boundaries(size):
    """Verify supported dimensions and boundary sizes survive request preparation."""
    payload, _ = ImageAPI.prepare({"prompt": {"text": "cat"}, "size": size})
    assert payload["size"] == size


def test_size_is_reassembled_from_integer_dimensions():
    """Verify dimensions are parsed as integers before rebuilding the size string."""
    payload, _ = ImageAPI.prepare({"prompt": {"text": "cat"}, "size": "0768x2304"})
    assert payload["size"] == "768x2304"


@pytest.mark.parametrize(
    "options",
    [
        {"size": "1025x1024"},
        {"size": "1920x1080"},
        {"size": "512x512"},
        {"size": "4096x2048"},
        {"size": "3840x3840"},
        {"size": "2048x640"},
        {"size": "0x1024"},
        {"size": "1024x0"},
        {"size": "1024x1025"},
        {"size": {"width": 1024}},
        {"size": {"width": 1024, "height": 1024}},
        {"size": "1024"},
        {"size": "1024X1024"},
        {"size": "1024x1024x1024"},
        {"size": "1024.0x1024"},
        {"size": "-1024x1024"},
        {"size": "auto"},
        {"size": ""},
        {"size": 1024},
        {"number_of_images": 0},
        {"number_of_images": 6},
        {"number_of_images": True},
        {"number_of_images": 1.5},
        {"model": "gpt-image-1"},
        {"quality": "auto"},
        {"complexity": "max"},
        {"background_transparency": "auto"},
        {"prompt": {"text": "   "}},
        {"prompt": {"text": "a" * 32001}},
        {"prompt": "cat"},
        {"prompt": {"text": "cat", "images": [0]}},
        {"prompt": {"text": "cat", "images": [True]}},
        {"prompt": {"text": "cat", "images": [1, 1]}},
        {"prompt": {"text": "cat", "images": list(range(1, 18))}},
        {"prompt": {"text": "cat", "images": ["/etc/passwd"]}},
        {"action": "edit"},
        {"mask": "file.png"},
        {"stream": True},
        {"moderation": "auto"},
        {"output_compression": 50},
    ],
)
def test_reject_invalid_or_unsupported_options(options):
    """Verify malformed, out-of-range, and unsupported arguments are rejected."""
    with pytest.raises(ImageGenerationError):
        ImageAPI.prepare({"prompt": {"text": "cat"}, **options})


def test_null_options_use_defaults():
    """Verify explicit null options behave like omitted optional arguments."""
    default, _ = ImageAPI.prepare({"prompt": {"text": "cat"}})
    options = dict.fromkeys(set(api_module.INPUT_SCHEMA["properties"]) - {"prompt"})
    actual, _ = ImageAPI.prepare({"prompt": {"text": "cat"}, **options})
    assert actual == default


@pytest.mark.asyncio
@pytest.mark.parametrize("with_reference", [False, True])
async def test_api_contract_and_original_bytes(api, tmp_path, webp, with_reference):
    """Verify generation and edit requests preserve the API contract and output bytes."""
    encoded = base64.b64encode(webp).decode()
    requests = []

    def respond(request):
        """Capture the request and return two copies of the WebP fixture."""
        requests.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": encoded}] * 2})

    api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/", transport=httpx.MockTransport(respond)
    )
    payload, _ = api.prepare(
        {
            "prompt": {"text": "cat"},
            "model": "sunburst",
            "quality": "max",
            "complexity": "low",
            "background_transparency": "transparent",
            "number_of_images": 2,
        }
    )
    try:
        paths = await api.generate(
            payload, [encoded] if with_reference else [], tmp_path
        )
        assert len(requests) == 1
        request = requests[0]
        assert request.url.path == (
            "/v1/images/edits" if with_reference else "/v1/images/generations"
        )
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["partial_images"] == 0
        assert body["output_compression"] == 100
        assert body["moderation"] == "low"
        assert body["model"] == "gpt-image-2.6-sunburst"
        assert body["quality"] == "max"
        assert body["background"] == "transparent"
        assert "action" not in body and "mask" not in body
        if with_reference:
            assert body["images"] == [
                {"image_url": "data:image/webp;base64," + encoded}
            ]
        else:
            assert "images" not in body
        assert len(set(paths)) == 2
        assert all(
            path.read_bytes() == webp and path.suffix == ".webp" for path in paths
        )
    finally:
        client = api.client
        await api.close()
        assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 400, 401, 403, 404, 429, 500, 503])
async def test_errors_do_not_leak_response_or_retry(api, tmp_path, status):
    """Verify HTTP errors report status without exposing response bodies or retrying."""
    attempts = []

    def respond(request):
        """Record each attempt and return an error containing private upstream content."""
        attempts.append(request)
        return httpx.Response(
            status,
            text="secret upstream content",
            headers={"location": "https://elsewhere.invalid"},
        )

    api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/", transport=httpx.MockTransport(respond)
    )
    payload, _ = api.prepare({"prompt": {"text": "cat"}})
    try:
        with pytest.raises(ImageGenerationError) as error:
            await api.generate(payload, [], tmp_path)
        assert str(status) in str(error.value)
        assert "secret" not in str(error.value)
        assert len(attempts) == 1
    finally:
        await api.close()


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"data": []},
        {"data": [None]},
        {"data": [{"b64_json": "invalid-base64!"}]},
        {"data": [{"url": "https://example.invalid/x"}]},
    ],
)
def test_invalid_responses_leave_no_files(tmp_path, result):
    """Verify invalid API responses are rejected before creating output files."""
    with pytest.raises(ImageGenerationError):
        ImageAPI.save_images(result, tmp_path / "out", 1)
    assert not (tmp_path / "out").exists()


def test_invalid_second_image_does_not_save_first(tmp_path, webp):
    """Verify all returned images are validated before any image is written."""
    with pytest.raises(ImageGenerationError):
        ImageAPI.save_images(
            {"data": [{"b64_json": base64.b64encode(webp).decode()}, {}]},
            tmp_path / "out",
            2,
        )
    assert not (tmp_path / "out").exists()


def test_non_webp_output_is_rejected(tmp_path):
    """Verify PNG is rejected as generated output but accepted as a reference image."""
    output = io.BytesIO()
    Image.new("RGB", (16, 16)).save(output, "PNG")
    encoded = base64.b64encode(output.getvalue()).decode()
    with pytest.raises(ImageGenerationError):
        ImageAPI.save_images({"data": [{"b64_json": encoded}]}, tmp_path / "out", 1)
    assert ImageAPI.reference_payload([encoded])[0]["image_url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.asyncio
async def test_missing_key_has_actionable_error(api, tmp_path):
    """Verify missing credentials produce a configuration error before client creation."""
    payload, _ = api.prepare({"prompt": {"text": "cat"}})
    with pytest.raises(ImageGenerationError, match="api_key"):
        await api.generate(payload, [], tmp_path)
    assert api.client is None


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["config", "environment", "file", "local"])
async def test_key_sources(api, tmp_path, monkeypatch, webp, source):
    """Verify API keys are loaded from each supported source with expected precedence."""
    key = "test-key-not-a-real-secret"
    if source == "config":
        api.config["api_key"] = key
        monkeypatch.setenv("OPENAI_API_KEY", "lower-priority")
    elif source == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", key)
    else:
        location = tmp_path / ".local" if source == "local" else tmp_path
        location.mkdir(exist_ok=True)
        (location / "OPENAI_API_KEY").write_text(key + "\n")
    captured = []
    client_type = httpx.AsyncClient

    def respond(request):
        """Capture the authorization header and return the WebP fixture."""
        captured.append(request.headers["authorization"])
        return httpx.Response(
            200, json={"data": [{"b64_json": base64.b64encode(webp).decode()}]}
        )

    def create_client(**kwargs):
        """Create an HTTP client using the test transport instead of network requests."""
        return client_type(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(api_module.httpx, "AsyncClient", create_client)
    payload, _ = api.prepare({"prompt": {"text": "cat"}})
    try:
        await api.generate(payload, [], tmp_path / "out")
        assert captured == ["Bearer " + key]
    finally:
        await api.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.invalid",
        "/v1",
        "https://x.invalid?secret=1",
        "https://x.invalid/#fragment",
        "https://user:secret@x.invalid",
        "https://x.invalid:badport",
    ],
)
async def test_invalid_base_url_is_safely_reported(api, tmp_path, base_url):
    """Verify invalid API roots are rejected without exposing sensitive URL contents."""
    api.config.update(api_key="test-key", base_url=base_url)
    payload, _ = api.prepare({"prompt": {"text": "cat"}})
    with pytest.raises(ImageGenerationError, match="base_url") as error:
        await api.generate(payload, [], tmp_path)
    assert "secret" not in str(error.value)
    assert api.client is None


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, "invalid", float("inf"), float("nan")])
async def test_invalid_timeout_prevents_request(api, tmp_path, timeout):
    """Verify invalid timeouts prevent HTTP client creation and API requests."""
    api.config.update(api_key="test-key", timeout_seconds=timeout)
    payload, _ = api.prepare({"prompt": {"text": "cat"}})
    with pytest.raises(ImageGenerationError, match="timeout_seconds"):
        await api.generate(payload, [], tmp_path)
    assert api.client is None
