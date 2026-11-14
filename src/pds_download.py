from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

from src.data_loader import parse_envi_header

ALLOWED_PDS_HOSTS = {"pds-geosciences.wustl.edu"}
GITHUB_RELEASE_HOSTS = {
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
    "github-production-release-asset-2e65be.s3.amazonaws.com",
}
GITHUB_RELEASE_CDN_HOSTS = GITHUB_RELEASE_HOSTS - {"github.com"}
MAX_IMG_BYTES = 350 * 1024 * 1024  # allow MEG128, reject unexpectedly huge downloads
MAX_METADATA_BYTES = 5 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "MarsBaseScout/0.2 (educational planetary-data viewer)"


def allowed_data_hosts() -> set[str]:
    """Allow PDS, GitHub Releases, and explicitly configured HTTPS hosts."""
    hosts = set(ALLOWED_PDS_HOSTS | GITHUB_RELEASE_HOSTS)
    for entry in os.environ.get("MARS_DATA_ALLOWED_HOSTS", "").split(","):
        host = entry.strip().lower().rstrip(".")
        if not host or "://" in host or "/" in host or "*" in host:
            continue
        try:
            import ipaddress
            ipaddress.ip_address(host)
        except ValueError:
            labels = host.split(".")
            valid_label = lambda label: (
                bool(label)
                and label[0].isalnum()
                and label[-1].isalnum()
                and all(character.isascii() and (character.isalnum() or character == "-") for character in label)
            )
            if len(labels) >= 2 and len(labels[-1]) >= 2 and labels[-1].isalpha() and all(valid_label(label) for label in labels):
                hosts.add(host)
    return hosts


def _request_file_response(url: str, **kwargs) -> requests.Response:
    """Request a file, following only the known CDN redirects used by GitHub Releases."""
    current_url = url
    source = urlsplit(url)
    github_release = (
        (source.hostname or "").lower().rstrip(".") == "github.com"
        and "/releases/download/" in source.path
    )

    for _ in range(6):
        response = requests.get(current_url, allow_redirects=False, **kwargs)
        if not response.is_redirect:
            return response

        location = response.headers.get("Location")
        response.close()
        if not github_release or not location:
            raise ValueError("Only GitHub Release asset redirects are supported.")

        next_url = urljoin(current_url, location)
        target = urlsplit(next_url)
        target_host = (target.hostname or "").lower().rstrip(".")
        if (
            target.scheme != "https"
            or target_host not in GITHUB_RELEASE_CDN_HOSTS
            or target.username
            or target.password
            or target.port not in (None, 443)
        ):
            raise ValueError("GitHub Release redirected to an untrusted download host.")
        current_url = next_url

    raise ValueError("GitHub Release used too many redirects.")


def _validated_img_url(url: str) -> str:
    value = (url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme != "https":
        raise ValueError("Güvenlik için yalnızca HTTPS PDS adresleri kabul ediliyor.")
    if (parsed.hostname or "").lower().rstrip(".") not in allowed_data_hosts():
        raise ValueError(
            "IMG URL must use PDS or a configured direct HTTPS data host."
        )
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("URL kimlik bilgisi veya standart dışı port içeremez.")
    if parsed.query or parsed.fragment:
        raise ValueError("IMG adresinde sorgu parametresi veya # parçası bulunmamalı.")
    if not parsed.path.lower().endswith(".img"):
        raise ValueError("Lütfen doğrudan .img dosyasının HTTPS adresini gir.")
    return value


def _sibling_url(img_url: str, suffix: str) -> str:
    parsed = urlsplit(img_url)
    path = parsed.path[:-4] + suffix
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _check_response(response: requests.Response, url: str) -> None:
    final = urlsplit(response.url)
    if final.scheme != "https" or (final.hostname or "").lower().rstrip(".") not in allowed_data_hosts():
        raise ValueError("PDS sunucusu izin verilmeyen bir adrese yönlendirdi; indirme durduruldu.")
    if response.is_redirect:
        raise ValueError("PDS bağlantısı yönlendirme döndürdü. Tarayıcıdaki doğrudan HTTPS IMG adresini kullan.")
    if response.status_code != 200:
        raise ValueError(f"PDS sunucusu HTTP {response.status_code} yanıtı verdi: {url}")



def _download_small_file(url: str, optional: bool = False) -> bytes | None:
    try:
        with _request_file_response(
            url,
            stream=True,
            timeout=(15, 45),
            headers={"User-Agent": USER_AGENT},
        ) as response:

            # İsteğe bağlı dosya bulunamazsa veya erişilemezse devam et.
            if optional and response.status_code != 200:
                return None

            _check_response(response, url)

            length = response.headers.get("Content-Length")

            if length and int(length) > MAX_METADATA_BYTES:
                if optional:
                    return None
                raise ValueError(
                    "HDR metadata dosyası beklenenden büyük; indirme reddedildi."
                )

            chunks = []
            total = 0

            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue

                total += len(chunk)

                if total > MAX_METADATA_BYTES:
                    if optional:
                        return None

                    raise ValueError(
                        "HDR metadata dosyası 5 MiB sınırını aşıyor."
                    )

                chunks.append(chunk)

            if not chunks:
                if optional:
                    return None

                raise ValueError("PDS HDR metadata dosyası boş döndü.")

            return b"".join(chunks)

    except requests.RequestException as exc:
        if optional:
            return None

        raise RuntimeError(
            f"PDS metadata dosyası indirilemedi: {exc}"
        ) from exc

def download_pds_product(
    img_url: str,
    progress_callback=None,
    temp_dir: str | None = None,
) -> tuple[str, bytes, bytes | None, str]:
    """Download sibling HDR/LBL and stream IMG to a temporary file.

    Returns (img_path, hdr_bytes, lbl_bytes, normalized_img_url).
    """
    img_url = _validated_img_url(img_url)
    hdr_url = _sibling_url(img_url, ".hdr")
    lbl_url = _sibling_url(img_url, ".lbl")

    hdr_bytes = _download_small_file(hdr_url, optional=False)
    lbl_bytes = _download_small_file(lbl_url, optional=True)
    header = parse_envi_header(hdr_bytes)
    if header["bands"] != 1:
        raise ValueError("Bu uygulama şu anda tek bantlı MOLA DEM ürünlerini destekliyor.")

    expected_minimum = (
        header["header_offset"]
        + header["samples"] * header["lines"] * header["dtype"].itemsize
    )
    if expected_minimum > MAX_IMG_BYTES:
        raise ValueError(
            f"HDR {expected_minimum / (1024 ** 2):.1f} MiB büyüklüğünde IMG bekliyor; "
            f"uygulama sınırı {MAX_IMG_BYTES / (1024 ** 2):.0f} MiB."
        )

    fd, img_path = tempfile.mkstemp(prefix="mars_mola_", suffix=".img", dir=temp_dir)
    downloaded = 0
    try:
        with os.fdopen(fd, "wb") as output:
            with _request_file_response(
                img_url,
                stream=True,
                timeout=(20, 90),
                headers={"User-Agent": USER_AGENT},
            ) as response:
                _check_response(response, img_url)
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise ValueError("PDS IMG adresi raster yerine HTML sayfası döndürdü.")
                content_length_raw = response.headers.get("Content-Length")
                content_length = int(content_length_raw) if content_length_raw else None
                if content_length is not None and content_length > MAX_IMG_BYTES:
                    raise ValueError(
                        f"IMG dosyası {content_length / (1024 ** 2):.1f} MiB; "
                        f"uygulama sınırı {MAX_IMG_BYTES / (1024 ** 2):.0f} MiB."
                    )
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_IMG_BYTES:
                        raise ValueError("IMG indirmesi 350 MiB sınırını aştı; işlem durduruldu.")
                    output.write(chunk)
                    if progress_callback:
                        progress_callback(downloaded, content_length)

        if downloaded < expected_minimum:
            raise ValueError(
                f"IMG indirmesi eksik: HDR en az {expected_minimum:,} bayt bekliyor, "
                f"{downloaded:,} bayt indirildi."
            )
        return img_path, hdr_bytes, lbl_bytes, img_url
    except requests.RequestException as exc:
        raise RuntimeError(f"MOLA IMG dosyası indirilemedi: {exc}") from exc
    except Exception:
        try:
            os.remove(img_path)
        except OSError:
            pass
        raise


PDS_ASSET_LIMITS = {
    ".tif": 500 * 1024 * 1024,
    ".tiff": 500 * 1024 * 1024,
    ".nc": 140 * 1024 * 1024,
    ".txt": 20 * 1024 * 1024,
}


def download_pds_asset(url: str, allowed_extensions: tuple[str, ...], temp_dir: str | None = None) -> tuple[str, str]:
    """Stream a supported non-IMG PDS asset to a bounded temporary file."""
    value = (url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme != "https" or (parsed.hostname or "").lower().rstrip(".") not in allowed_data_hosts():
        raise ValueError("Yaln\u0131zca yap\u0131land\u0131r\u0131lm\u0131\u015f PDS sunucusundaki do\u011frudan HTTPS dosyalar\u0131 destekleniyor.")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("PDS URL\u2019leri kimlik bilgisi veya standart d\u0131\u015f\u0131 port i\u00e7ermemeli.")
    if parsed.query or parsed.fragment:
        raise ValueError("PDS dosya URL\u2019lerinde sorgu parametresi veya par\u00e7a bulunmamal\u0131.")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in allowed_extensions or suffix not in PDS_ASSET_LIMITS:
        raise ValueError(f"Do\u011frudan PDS dosyas\u0131 i\u00e7in desteklenen uzant\u0131lardan birini se\u00e7in: {', '.join(allowed_extensions)}.")

    normalized_url = value
    limit = PDS_ASSET_LIMITS[suffix]
    fd, asset_path = tempfile.mkstemp(prefix="mars_pds_", suffix=suffix, dir=temp_dir)
    downloaded = 0
    try:
        with os.fdopen(fd, "wb") as output:
            with _request_file_response(
                normalized_url,
                stream=True,
                timeout=(20, 90),
                headers={"User-Agent": USER_AGENT},
            ) as response:
                _check_response(response, normalized_url)
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("PDS URL veri dosyas\u0131 yerine HTML sayfas\u0131 d\u00f6nd\u00fcrd\u00fc.")
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > limit:
                    raise ValueError(f"PDS dosyas\u0131 beklenen boyut s\u0131n\u0131r\u0131n\u0131 a\u015f\u0131yor: {limit // (1024 * 1024)} MiB indirme s\u0131n\u0131r\u0131.")
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > limit:
                        raise ValueError(f"PDS dosyas\u0131 beklenen boyut s\u0131n\u0131r\u0131n\u0131 a\u015f\u0131yor: {limit // (1024 * 1024)} MiB indirme s\u0131n\u0131r\u0131.")
                    output.write(chunk)
        if downloaded == 0:
            raise ValueError("PDS veri dosyas\u0131 bo\u015f geldi.")
        return asset_path, normalized_url
    except requests.RequestException as exc:
        try:
            os.remove(asset_path)
        except OSError:
            pass
        raise RuntimeError(f"PDS veri dosyas\u0131 indirilemedi: {exc}") from exc
    except Exception:
        try:
            os.remove(asset_path)
        except OSError:
            pass
        raise
