from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from src.data_loader import parse_envi_header

ALLOWED_PDS_HOSTS = {"pds-geosciences.wustl.edu"}
MAX_IMG_BYTES = 350 * 1024 * 1024  # allow MEG128, reject unexpectedly huge downloads
MAX_METADATA_BYTES = 5 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "MarsBaseScout/0.2 (educational planetary-data viewer)"


def _validated_img_url(url: str) -> str:
    value = (url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme != "https":
        raise ValueError("Güvenlik için yalnızca HTTPS PDS adresleri kabul ediliyor.")
    if parsed.hostname not in ALLOWED_PDS_HOSTS:
        raise ValueError(
            "Şimdilik yalnızca pds-geosciences.wustl.edu arşivindeki doğrudan IMG bağlantıları destekleniyor."
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
    if final.scheme != "https" or final.hostname not in ALLOWED_PDS_HOSTS:
        raise ValueError("PDS sunucusu izin verilmeyen bir adrese yönlendirdi; indirme durduruldu.")
    if response.is_redirect:
        raise ValueError("PDS bağlantısı yönlendirme döndürdü. Tarayıcıdaki doğrudan HTTPS IMG adresini kullan.")
    if response.status_code != 200:
        raise ValueError(f"PDS sunucusu HTTP {response.status_code} yanıtı verdi: {url}")


def _download_small_file(url: str, optional: bool = False) -> bytes | None:
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 45),
            allow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        ) as response:
            if optional and response.status_code == 404:
                return None
            _check_response(response, url)
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_METADATA_BYTES:
                raise ValueError("HDR/LBL metadata dosyası beklenenden büyük; indirme reddedildi.")
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_METADATA_BYTES:
                    raise ValueError("HDR/LBL metadata dosyası 5 MiB sınırını aşıyor.")
                chunks.append(chunk)
            if not chunks:
                raise ValueError("PDS metadata dosyası boş döndü.")
            return b"".join(chunks)
    except requests.RequestException as exc:
        raise RuntimeError(f"PDS metadata dosyası indirilemedi: {exc}") from exc


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
            with requests.get(
                img_url,
                stream=True,
                timeout=(20, 90),
                allow_redirects=False,
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
