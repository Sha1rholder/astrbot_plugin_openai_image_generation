import base64
import io
import json

import httpx
import pytest
import yaml
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
    payload, refs = ImageAPI.prepare(
        {"description": "A cat", "prompt": {"text": " Draw a cat "}}
    )
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
    payload, _ = ImageAPI.prepare(
        {"description": "A cat", "prompt": {"text": "cat"}, "size": size}
    )
    assert payload["size"] == size


def test_size_is_reassembled_from_integer_dimensions():
    """Verify dimensions are parsed as integers before rebuilding the size string."""
    payload, _ = ImageAPI.prepare(
        {"description": "A cat", "prompt": {"text": "cat"}, "size": "0768x2304"}
    )
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
        {"description": "A cat", "prompt": {"text": "   "}},
        {"description": "A cat", "prompt": {"text": "a" * 32001}},
        {"prompt": "cat"},
        {"description": "A cat", "prompt": {"text": "cat", "images": [0]}},
        {"description": "A cat", "prompt": {"text": "cat", "images": [True]}},
        {"description": "A cat", "prompt": {"text": "cat", "images": [1, 1]}},
        {
            "description": "A cat",
            "prompt": {"text": "cat", "images": list(range(1, 18))},
        },
        {"description": "A cat", "prompt": {"text": "cat", "images": ["/etc/passwd"]}},
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
        ImageAPI.prepare({"description": "A cat", "prompt": {"text": "cat"}, **options})


def test_null_options_use_defaults():
    """Verify explicit null options behave like omitted optional arguments."""
    default, _ = ImageAPI.prepare({"description": "A cat", "prompt": {"text": "cat"}})
    options = dict.fromkeys(
        set(api_module.INPUT_SCHEMA["properties"]) - {"prompt", "description"}
    )
    actual, _ = ImageAPI.prepare(
        {"description": "A cat", "prompt": {"text": "cat"}, **options}
    )
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
    arguments = {
        "description": "A cat",
        "prompt": {"text": "  cat  ", "images": [1] if with_reference else []},
        "model": "sunburst",
        "quality": "max",
        "complexity": "low",
        "size": None,
        "background_transparency": "transparent",
        "number_of_images": 2,
    }
    (api.plugin_dir / "metadata.yaml").write_text("version: v9.8.7\n")
    payload, _ = api.prepare(arguments)
    try:
        paths = await api.generate(
            payload,
            [encoded] if with_reference else [],
            tmp_path,
            arguments,
        )
        archive_text = (paths[0].parent / "request.yaml").read_text()
        assert archive_text.splitlines()[0] == (
            "# yaml-language-server: $schema=" + api_module.REQUEST_SCHEMA["$id"]
        )
        api_module.jsonschema.validate(
            yaml.safe_load(archive_text), api_module.REQUEST_SCHEMA
        )
        assert yaml.safe_load(archive_text) == {
            "plugin_version": "v9.8.7",
            "request": arguments,
        }
        assert len(requests) == 1
        request = requests[0]
        assert request.url.path == (
            "/v1/images/edits" if with_reference else "/v1/images/generations"
        )
        body = json.loads(request.content)
        assert "description" not in body
        assert "plugin_version" not in body
        assert body["prompt"] == "cat"
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
    payload, _ = api.prepare({"description": "A cat", "prompt": {"text": "cat"}})
    try:
        with pytest.raises(ImageGenerationError) as error:
            await api.generate(
                payload,
                [],
                tmp_path,
                {"description": "A cat", "prompt": {"text": "cat"}},
            )
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
        ImageAPI.save_images(
            result,
            tmp_path / "out",
            1,
            {"description": "A cat", "prompt": {"text": "cat"}},
            "v0.1.0",
        )
    assert not (tmp_path / "out").exists()


def test_invalid_second_image_does_not_save_first(tmp_path, webp):
    """Verify all returned images are validated before any image is written."""
    with pytest.raises(ImageGenerationError):
        ImageAPI.save_images(
            {"data": [{"b64_json": base64.b64encode(webp).decode()}, {}]},
            tmp_path / "out",
            2,
            {"description": "A cat", "prompt": {"text": "cat"}},
            "v0.1.0",
        )
    assert not (tmp_path / "out").exists()


def test_non_webp_output_is_rejected(tmp_path):
    """Verify PNG is rejected as generated output but accepted as a reference image."""
    output = io.BytesIO()
    Image.new("RGB", (16, 16)).save(output, "PNG")
    encoded = base64.b64encode(output.getvalue()).decode()
    with pytest.raises(ImageGenerationError):
        ImageAPI.save_images(
            {"data": [{"b64_json": encoded}]},
            tmp_path / "out",
            1,
            {"description": "A cat", "prompt": {"text": "cat"}},
            "v0.1.0",
        )
    assert ImageAPI.reference_payload([encoded])[0]["image_url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.asyncio
async def test_missing_key_has_actionable_error(api, tmp_path):
    """Verify missing credentials produce a configuration error before client creation."""
    payload, _ = api.prepare({"description": "A cat", "prompt": {"text": "cat"}})
    with pytest.raises(ImageGenerationError, match="api_key"):
        await api.generate(
            payload, [], tmp_path, {"description": "A cat", "prompt": {"text": "cat"}}
        )
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
    payload, _ = api.prepare({"description": "A cat", "prompt": {"text": "cat"}})
    try:
        await api.generate(
            payload,
            [],
            tmp_path / "out",
            {"description": "A cat", "prompt": {"text": "cat"}},
        )
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
    payload, _ = api.prepare({"description": "A cat", "prompt": {"text": "cat"}})
    with pytest.raises(ImageGenerationError, match="base_url") as error:
        await api.generate(
            payload, [], tmp_path, {"description": "A cat", "prompt": {"text": "cat"}}
        )
    assert "secret" not in str(error.value)
    assert api.client is None


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, "invalid", float("inf"), float("nan")])
async def test_invalid_timeout_prevents_request(api, tmp_path, timeout):
    """Verify invalid timeouts prevent HTTP client creation and API requests."""
    api.config.update(api_key="test-key", timeout_seconds=timeout)
    payload, _ = api.prepare({"description": "A cat", "prompt": {"text": "cat"}})
    with pytest.raises(ImageGenerationError, match="timeout_seconds"):
        await api.generate(
            payload, [], tmp_path, {"description": "A cat", "prompt": {"text": "cat"}}
        )
    assert api.client is None


@pytest.mark.parametrize("description", [None, "", "   ", 123, "x" * 1001])
def test_invalid_description(description):
    with pytest.raises(ImageGenerationError):
        ImageAPI.prepare({"prompt": {"text": "cat"}, "description": description})


def test_description_required():
    with pytest.raises(ImageGenerationError):
        ImageAPI.prepare({"prompt": {"text": "cat"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, True, "10", None, 1.5])
async def test_invalid_retention_prevents_request(api, tmp_path, limit):
    api.config["max_saved_requests"] = limit
    arguments = {"prompt": {"text": "cat"}, "description": "A cat"}
    payload, _ = api.prepare(arguments)
    with pytest.raises(ImageGenerationError, match="max_saved_requests"):
        await api.generate(payload, [], tmp_path, arguments)
    assert api.client is None


def test_request_archive_and_numeric_retention(tmp_path, webp):
    import shutil

    arguments = {
        "prompt": {"text": "  橘猫坐在窗边  ", "images": [1]},
        "description": "窗边的橘猫",
        "quality": None,
        "number_of_images": 2,
    }
    result = {"data": [{"b64_json": base64.b64encode(webp).decode()}] * 2}
    for number in range(12):
        paths = ImageAPI.save_images(result, tmp_path, 2, arguments, "v9.8.7")
        assert paths == [tmp_path / str(number) / f"{i}.webp" for i in range(2)]
        assert all(p.read_bytes() == webp for p in paths)
        assert (paths[0].parent / "description.txt").read_text() == arguments[
            "description"
        ]
        archive_text = (paths[0].parent / "request.yaml").read_text()
        assert archive_text.splitlines()[0] == (
            "# yaml-language-server: $schema=" + api_module.REQUEST_SCHEMA["$id"]
        )
        api_module.jsonschema.validate(
            yaml.safe_load(archive_text), api_module.REQUEST_SCHEMA
        )
        assert yaml.safe_load(archive_text) == {
            "plugin_version": "v9.8.7",
            "request": arguments,
        }
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == {
        str(i) for i in range(2, 12)
    }
    # Reducing the limit cleans all directories below the numeric cutoff.
    paths = ImageAPI.save_images(result, tmp_path, 2, arguments, "v9.8.7", 1)
    assert paths[0].parent.name == "12"
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == {"12"}
    # Clearing all request directories resets the next ID to zero.
    shutil.rmtree(paths[0].parent)
    paths = ImageAPI.save_images(result, tmp_path, 2, arguments, "v9.8.7", 1)
    assert paths[0].parent.name == "0"
    assert not (tmp_path / ".next_request_id").exists()


def test_storage_failure_keeps_previous_request(tmp_path, webp, monkeypatch):
    from pathlib import Path

    arguments = {"prompt": {"text": "cat"}, "description": "cat"}
    result = {"data": [{"b64_json": base64.b64encode(webp).decode()}]}
    ImageAPI.save_images(result, tmp_path, 1, arguments, "v1", 1)
    original = Path.write_text

    def fail_metadata(path, *args, **kwargs):
        if path.name == "request.yaml":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(OSError):
        ImageAPI.save_images(result, tmp_path, 1, arguments, "v1", 1)
    assert {p.name for p in tmp_path.iterdir()} == {"0"}
    assert (tmp_path / "0" / "0.webp").read_bytes() == webp


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
    ImageAPI.prune_requests(output, 1)
    assert not (output / "1").exists()
    assert (outside / "keep").read_text() == "keep"
    assert {p.name for p in output.iterdir()} == {
        "0",
        "2",
        "02",
        "old-uuid",
        "old.webp",
    }


@pytest.mark.asyncio
async def test_missing_metadata_prevents_request(api, tmp_path):
    (tmp_path / "metadata.yaml").unlink()
    arguments = {"prompt": {"text": "cat"}, "description": "cat"}
    payload, _ = api.prepare(arguments)
    with pytest.raises(ImageGenerationError, match="metadata.yaml"):
        await api.generate(payload, [], tmp_path, arguments)
    assert api.client is None


def test_next_request_id_scans_only_numeric_directories(tmp_path):
    assert ImageAPI.next_request_id(tmp_path / "missing") == 0
    assert ImageAPI.next_request_id(tmp_path) == 0
    (tmp_path / "0").mkdir()
    assert ImageAPI.next_request_id(tmp_path) == 1
    for name in ["2", "9", "10", "-20", "1.5", "notes"]:
        (tmp_path / name).mkdir()
    (tmp_path / "999").write_text("not a directory")
    (tmp_path / "1000").symlink_to(tmp_path / "2", target_is_directory=True)
    (tmp_path / ".next_request_id").write_text("5000")
    assert ImageAPI.next_request_id(tmp_path) == 11
    (tmp_path / "10").rmdir()
    assert ImageAPI.next_request_id(tmp_path) == 10


@pytest.mark.asyncio
async def test_number_is_selected_before_http_request(api, tmp_path, webp):
    output = tmp_path / "images"
    output.mkdir()
    (output / "2").mkdir()
    arguments = {"prompt": {"text": "cat"}, "description": "cat"}
    payload, _ = api.prepare(arguments)

    def respond(_request):
        # A change during generation must not alter the already selected ID.
        (output / "7").mkdir()
        return httpx.Response(
            200, json={"data": [{"b64_json": base64.b64encode(webp).decode()}]}
        )

    api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/", transport=httpx.MockTransport(respond)
    )
    try:
        paths = await api.generate(payload, [], output, arguments)
        assert paths == [output / "3" / "0.webp"]
        assert not (output / ".next_request_id").exists()
    finally:
        await api.close()


def test_archive_schema_explains_tool_arguments_and_rejects_api_payload():
    api_module.jsonschema.Draft202012Validator.check_schema(api_module.REQUEST_SCHEMA)
    with pytest.raises(api_module.jsonschema.ValidationError):
        api_module.jsonschema.validate(
            {
                "plugin_version": "v0.1.0",
                "request": {"prompt": "cat", "model": "gpt-image-2.6-flare", "n": 1},
            },
            api_module.REQUEST_SCHEMA,
        )
