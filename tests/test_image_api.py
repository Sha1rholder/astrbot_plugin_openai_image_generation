import base64
import io
import json
from unittest.mock import Mock

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
        directory = tmp_path / "0"
        assert (
            yaml.safe_load((directory / "request.yaml").read_text())["request"][
                "description"
            ]
            == "A cat"
        )
        assert {path.name for path in directory.iterdir()} == {"request.yaml"}
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
def test_invalid_responses_preserve_request(api, tmp_path, result):
    """Invalid API responses leave the original request without any output images."""
    directory = api.create_request(
        tmp_path / "out", {"description": "A cat", "prompt": {"text": "cat"}}
    )
    with pytest.raises(ImageGenerationError):
        ImageAPI.save_images(
            result,
            directory,
            1,
        )
    assert {path.name for path in directory.iterdir()} == {"request.yaml"}


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
        ImageAPI.save_images(
            {"data": [{"b64_json": encoded}]},
            tmp_path / "out",
            1,
        )
    assert ImageAPI.reference_payload([encoded])[0]["image_url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("key_file", [None, "OPENAI_API_KEY", ".local/OPENAI_API_KEY"])
async def test_missing_key_has_actionable_error(api, tmp_path, monkeypatch, key_file):
    """Verify missing credentials fail before client creation even when key files exist."""
    if key_file is not None:
        path = api.plugin_dir / key_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored-file-key\n", encoding="utf-8")
    api.config["api_key"] = " \t"
    monkeypatch.setenv("OPENAI_API_KEY", " \t")
    create_client = Mock()
    monkeypatch.setattr(api_module.httpx, "AsyncClient", create_client)
    payload, _ = api.prepare({"description": "A cat", "prompt": {"text": "cat"}})
    with pytest.raises(ImageGenerationError, match="api_key") as error:
        await api.generate(
            payload, [], tmp_path, {"description": "A cat", "prompt": {"text": "cat"}}
        )
    assert "OPENAI_API_KEY environment variable" in str(error.value)
    assert "key file" not in str(error.value)
    create_client.assert_not_called()
    assert api.client is None


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["config", "environment"])
async def test_key_sources(api, tmp_path, monkeypatch, webp, source):
    """Verify API keys are loaded from each supported source with expected precedence."""
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
@pytest.mark.parametrize("limit", [0, -1, True, "10", None, 1.5, 1.0, float("inf"), float("nan")])
async def test_invalid_retention_prevents_request(api, tmp_path, limit):
    api.config["max_storage_gb"] = limit
    arguments = {"prompt": {"text": "cat"}, "description": "A cat"}
    payload, _ = api.prepare(arguments)
    with pytest.raises(ImageGenerationError, match="max_storage_gb"):
        await api.generate(payload, [], tmp_path, arguments)
    assert api.client is None


@pytest.mark.asyncio
async def test_request_archive_and_capacity_retention(api, tmp_path, webp, monkeypatch):
    import shutil

    arguments = {
        "prompt": {"text": "  橘猫坐在窗边  ", "images": [1]},
        "description": "窗边的橘猫",
        "quality": None,
        "number_of_images": 2,
    }
    result = {"data": [{"b64_json": base64.b64encode(webp).decode()}] * 2}
    (api.plugin_dir / "metadata.yaml").write_text("version: v9.8.7\n")
    output = tmp_path / "images"
    api.client = httpx.AsyncClient(
        base_url="https://example.invalid/v1/",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=result)),
    )
    payload, _ = api.prepare(arguments)
    for number in range(12):
        paths = await api.generate(payload, [], output, arguments)
        assert paths == [output / str(number) / f"{i}.webp" for i in range(2)]
        assert all(p.read_bytes() == webp for p in paths)
        assert not (paths[0].parent / "description.txt").exists()
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
    assert {p.name for p in output.iterdir() if p.is_dir()} == {
        str(i) for i in range(12)
    }
    # Reducing the limit cleans old archives before the next HTTP request.
    archive_size = (output / "11" / "request.yaml").stat().st_size
    monkeypatch.setattr(api, "storage_limit", lambda: archive_size + 1)
    paths = await api.generate(payload, [], output, arguments)
    assert paths[0].parent.name == "12"
    assert {p.name for p in output.iterdir() if p.is_dir()} == {"12"}
    # Clearing all request directories resets the next ID to zero.
    shutil.rmtree(paths[0].parent)
    paths = await api.generate(payload, [], output, arguments)
    assert paths[0].parent.name == "0"
    assert not (output / ".next_request_id").exists()

    await api.close()


def test_storage_failure_keeps_previous_request(api, tmp_path, webp, monkeypatch):
    from pathlib import Path

    arguments = {"prompt": {"text": "cat"}, "description": "cat"}
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


@pytest.mark.parametrize("operation", ["write_bytes", "replace"])
def test_image_storage_failure_preserves_request(
    api, tmp_path, webp, monkeypatch, operation
):
    from pathlib import Path

    arguments = {"prompt": {"text": "cat"}, "description": "cat"}
    directory = api.create_request(tmp_path / "images", arguments)
    archive = (directory / "request.yaml").read_bytes()
    original = getattr(Path, operation)

    def fail_second_image(path, *args, **kwargs):
        if path.name == "1.webp":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, fail_second_image)
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
        assert (
            yaml.safe_load((output / "3" / "request.yaml").read_text())["request"]
            == arguments
        )
        assert not (output / "3" / "0.webp").exists()
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


@pytest.mark.parametrize("limit,remaining", [(100, {"0", "2", "10"}), (99, {"2", "10"}), (60, {"10"})])
def test_capacity_cleanup_order_and_boundary(tmp_path, limit, remaining):
    for name, size in [("10", 30), ("2", 30), ("0", 40)]:
        directory = tmp_path / name / "nested"
        directory.mkdir(parents=True)
        (directory / "image.webp").write_bytes(b"x" * size)
    ImageAPI.prune_requests(tmp_path, limit, 11)
    assert {p.name for p in tmp_path.iterdir()} == remaining


@pytest.mark.asyncio
async def test_cleanup_precedes_http_and_keeps_current_and_queued(api, tmp_path, monkeypatch):
    arguments = {"prompt": {"text": "cat"}, "description": "cat"}
    output = tmp_path / "images"
    old = api.create_request(output, arguments)
    (old / "large.webp").write_bytes(b"x" * 10000)
    current = api.create_request(output, arguments)
    queued = api.create_request(output, arguments)
    monkeypatch.setattr(api, "storage_limit", lambda: ImageAPI.storage_size(current) * 2 + 1)
    calls = []

    def respond(request):
        assert not old.exists()
        assert (current / "request.yaml").exists()
        assert (queued / "request.yaml").exists()
        calls.append(request)
        return httpx.Response(500)

    api.client = httpx.AsyncClient(base_url="https://example.invalid/v1/", transport=httpx.MockTransport(respond))
    payload, _ = api.prepare(arguments)
    try:
        with pytest.raises(ImageGenerationError, match="HTTP 500"):
            await api.generate(payload, [], output, arguments, current)
        assert len(calls) == 1
        monkeypatch.setattr(api, "storage_limit", lambda: 1)
        with pytest.raises(ImageGenerationError, match="No Image API request"):
            await api.generate(payload, [], output, arguments, current)
        assert len(calls) == 1
        assert current.exists() and queued.exists()
    finally:
        await api.close()


def test_cleanup_failure_prevents_deletion_continuation(tmp_path, monkeypatch):
    directory = tmp_path / "0"
    directory.mkdir()
    (directory / "image").write_bytes(b"xx")

    def fail(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(api_module.shutil, "rmtree", fail)
    with pytest.raises(PermissionError):
        ImageAPI.prune_requests(tmp_path, 1, 1)
    assert directory.exists()


def test_storage_limit_defaults_to_ten_gb(api):
    assert api.storage_limit() == 10_000_000_000


@pytest.mark.parametrize("limit", [1, 10, 25])
def test_storage_limit_accepts_positive_integers(api, limit):
    api.config["max_storage_gb"] = limit
    assert api.storage_limit() == limit * 1_000_000_000
