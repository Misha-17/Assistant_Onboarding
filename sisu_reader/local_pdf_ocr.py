"""Optional bounded CPU OCR for PDF pages without native extractable text.

Staged outside the live SISU package. Windows OCR output is an uncertain
transcription, not a certified source quote. Native text pages never invoke OCR.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

CONTRACT = "windows-pdf-ocr-v1"
WORKER = Path(__file__).with_name("windows_ocr.ps1")


@dataclass(frozen=True)
class OcrLimits:
    max_file_bytes: int = 50_000_000
    max_pages: int = 20
    max_page_pixels: int = 8_000_000
    max_total_pixels: int = 50_000_000
    max_dimension: int = 4000
    dpi: int = 180
    timeout_s: float = 30.0
    page_timeout_s: float = 8.0
    max_text_chars: int = 200_000
    max_regions: int = 5000

    def __post_init__(self):
        bounds = {"max_file_bytes": (1, 100_000_000), "max_pages": (1, 64),
                  "max_page_pixels": (1000, 16_000_000), "max_total_pixels": (1000, 100_000_000),
                  "max_dimension": (32, 10000), "dpi": (72, 300),
                  "max_text_chars": (1, 1_000_000), "max_regions": (1, 10000)}
        for key, (minimum, maximum) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"Invalid OCR limit: {key}")
        for key, maximum in (("timeout_s", 120), ("page_timeout_s", 60)):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0.1 <= value <= maximum:
                raise ValueError(f"Invalid OCR time limit: {key}")


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _blank(path, limits, language):
    return {"schema_version": 1, "contract": CONTRACT, "source_path": str(path),
            "source_sha256": None, "language_requested": language,
            "backend": "windows-media-ocr", "backend_script_sha256": digest(WORKER),
            "limits": asdict(limits), "pages": [], "flags": [], "total_pages": None,
            "processing_complete": False, "text_completeness": "not_certified",
            "ocr_calls": 0, "rendered_pixels": 0, "elapsed_s": 0.0}


def render_dimensions(width_points, height_points, limits):
    if not all(type(value) in (int, float) and math.isfinite(value) and value > 0
               for value in (width_points, height_points)):
        raise ValueError("Invalid PDF page dimensions")
    scale = min(limits.dpi / 72, limits.max_dimension / width_points,
                limits.max_dimension / height_points,
                math.sqrt(limits.max_page_pixels / (width_points * height_points)))
    # Leave room for integer raster rounding while enforcing the preallocation cap.
    width, height = math.ceil(width_points * scale), math.ceil(height_points * scale)
    while width * height > limits.max_page_pixels or max(width, height) > limits.max_dimension:
        scale *= 0.999
        width, height = math.ceil(width_points * scale), math.ceil(height_points * scale)
    if min(width, height) < 1:
        raise ValueError("Page cannot fit the raster limits")
    return scale, width, height


def recognize_windows(image_path, language, timeout_s):
    if os.name != "nt":
        raise RuntimeError("windows_ocr_unavailable")
    executable = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    environment = dict(os.environ, SISU_OCR_IMAGE_PATH=str(Path(image_path).resolve()),
                       SISU_OCR_LANGUAGE=language, SISU_OCR_TIMEOUT_MS=str(max(1, int(timeout_s * 1000))))
    # The command is fixed trusted code. File paths/language remain environment
    # data; document/OCR strings are never interpolated into a shell command.
    result = subprocess.run([str(executable), "-NoProfile", "-NonInteractive", "-Command",
                             WORKER.read_text(encoding="utf-8")],
                            capture_output=True, text=True, encoding="utf-8", timeout=timeout_s,
                            env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("windows_ocr_backend_error")
    if len(result.stdout) > 8_000_000:
        raise ValueError("ocr_backend_output_limit")
    value = json.loads(result.stdout)
    if value.get("schema_version") != 1 or value.get("backend") != "windows-media-ocr":
        raise ValueError("ocr_backend_contract_mismatch")
    return value


def _regions(raw, *, page_number, page_width, page_height, pixel_width, pixel_height, limits):
    if raw.get("image_width") != pixel_width or raw.get("image_height") != pixel_height:
        raise ValueError("ocr_raster_dimensions_mismatch")
    lines = raw.get("lines")
    if not isinstance(lines, list):
        raise ValueError("ocr_invalid_lines")
    regions, flags, characters = [], [], 0
    for line in lines:
        if len(regions) >= limits.max_regions:
            flags.append("ocr_region_limit")
            break
        text = line.get("text") if isinstance(line, dict) else None
        words = line.get("words") if isinstance(line, dict) else None
        if not isinstance(text, str) or not text.strip() or not isinstance(words, list) or not words:
            flags.append("ocr_invalid_region")
            continue
        if characters + len(text) > limits.max_text_chars:
            flags.append("ocr_text_limit")
            break
        boxes, word_records = [], []
        for word in words:
            box = word.get("box") if isinstance(word, dict) else None
            if (not isinstance(box, list) or len(box) != 4 or
                    not all(type(v) in (int, float) and math.isfinite(v) for v in box)
                    or not isinstance(word.get("text"), str)):
                raise ValueError("ocr_invalid_word_geometry")
            x, y, width, height = box
            if min(x, y) < 0 or min(width, height) <= 0 or x + width > pixel_width + 1 or y + height > pixel_height + 1:
                raise ValueError("ocr_word_outside_page")
            boxes.append((x, y, x + width, y + height))
            word_records.append({"text": word["text"], "box_pixels": box, "confidence": None})
        pixel_box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes)]
        point_box = [round(pixel_box[0] * page_width / pixel_width, 3),
                     round(pixel_box[1] * page_height / pixel_height, 3),
                     round(pixel_box[2] * page_width / pixel_width, 3),
                     round(pixel_box[3] * page_height / pixel_height, 3)]
        region_flags = ["ocr_text_unverified", "ocr_confidence_unavailable", "ocr_layout_unverified"]
        if "\ufffd" in text or any(ord(character) < 32 and character not in "\n\r\t" for character in text):
            region_flags.append("ocr_suspicious_characters")
        ordinal = len(regions) + 1
        regions.append({"region": ordinal, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "box_pixels": pixel_box, "box_pdf_points": point_box,
                        "coordinate_system": "rendered_page_top_left_pdf_points",
                        "locator": f"page {page_number}, OCR region {ordinal}, bbox=" + ",".join(map(str, point_box)),
                        "words": word_records, "confidence": None, "flags": region_flags})
        characters += len(text)
    return regions, flags


def spatial_text(regions):
    """Join only detected text, ordering vertical rows and then left-to-right.

    Horizontal alignment is inferred, never declared to be a verified table.
    Region records and original engine order remain available for inspection.
    """
    ordered = sorted(regions, key=lambda item: (item["box_pixels"][1], item["box_pixels"][0]))
    rows = []
    for region in ordered:
        box = region["box_pixels"]
        center, height = (box[1] + box[3]) / 2, box[3] - box[1]
        if rows and abs(center - rows[-1]["center"]) <= min(height, rows[-1]["height"]) * 0.5:
            rows[-1]["regions"].append(region)
        else:
            rows.append({"center": center, "height": height, "regions": [region]})
    return "\n".join("    ".join(item["text"] for item in sorted(row["regions"], key=lambda item: item["box_pixels"][0]))
                     for row in rows)


def _extract_worker(path, limits, language, output, *, recognizer=recognize_windows):
    started = time.monotonic()
    report = _blank(path, limits, language)
    report["source_sha256"] = digest(path)
    total_characters = total_regions = 0
    try:
        import pymupdf
        with pymupdf.open(path) as document:
            report["renderer_version"] = pymupdf.VersionBind
            report["total_pages"] = len(document)
            if document.needs_pass:
                report["flags"].append("ocr_encrypted_pdf_not_opened")
                return report
            with tempfile.TemporaryDirectory(prefix="sisu-ocr-raster-") as raster_dir:
                for page_index in range(min(len(document), limits.max_pages)):
                    remaining = limits.timeout_s - (time.monotonic() - started)
                    if remaining <= 0.1:
                        report["flags"].append("ocr_document_timeout")
                        break
                    page = document[page_index]
                    entry = {"page": page_index + 1, "status": "unprocessed", "regions": [], "flags": []}
                    report["pages"].append(entry)
                    if page.get_text("text").strip():
                        entry["status"] = "native_text_present_no_ocr"
                        _write(output, report)
                        continue
                    scale, width, height = render_dimensions(page.rect.width, page.rect.height, limits)
                    if report["rendered_pixels"] + width * height > limits.max_total_pixels:
                        entry["status"] = "pixel_budget_exhausted"
                        report["flags"].append("ocr_total_pixel_limit")
                        break
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csRGB)
                    if pixmap.width * pixmap.height > limits.max_page_pixels:
                        raise ValueError("ocr_actual_raster_exceeds_limit")
                    report["rendered_pixels"] += pixmap.width * pixmap.height
                    image_path = Path(raster_dir) / f"page-{page_index + 1}.png"
                    pixmap.save(image_path)
                    entry["raster"] = {"width": pixmap.width, "height": pixmap.height,
                                       "effective_dpi": scale * 72, "sha256": digest(image_path)}
                    del pixmap
                    page_started = time.monotonic()
                    try:
                        remaining = limits.timeout_s - (time.monotonic() - started)
                        if remaining <= 0.1:
                            raise TimeoutError("ocr_document_timeout")
                        report["ocr_calls"] += 1
                        raw = recognizer(image_path, language, min(limits.page_timeout_s, remaining))
                        remaining_limits = OcrLimits(**{**asdict(limits),
                            "max_text_chars": max(1, limits.max_text_chars - total_characters),
                            "max_regions": max(1, limits.max_regions - total_regions)})
                        regions, flags = _regions(raw, page_number=page_index + 1,
                            page_width=page.rect.width, page_height=page.rect.height,
                            pixel_width=entry["raster"]["width"], pixel_height=entry["raster"]["height"],
                            limits=remaining_limits)
                        entry.update(status="ocr_text_unverified" if regions else "ocr_no_text_found",
                                     regions=regions, flags=flags, text=spatial_text(regions),
                                     backend_version=raw.get("backend_version"), language=raw.get("language"),
                                     text_angle=raw.get("text_angle"), confidence=None)
                        total_characters += sum(len(region["text"]) for region in regions)
                        total_regions += len(regions)
                    except (subprocess.TimeoutExpired, TimeoutError):
                        entry["status"] = "ocr_timeout"
                        entry["flags"].append("ocr_page_timeout")
                    except Exception as exc:
                        entry["status"] = "ocr_error"
                        entry["flags"].append("ocr_backend_error:" + type(exc).__name__)
                    entry["ocr_wall_elapsed_s"] = time.monotonic() - page_started
                    _write(output, report)
                    if total_characters >= limits.max_text_chars or total_regions >= limits.max_regions:
                        report["flags"].append("ocr_document_output_limit")
                        break
            if len(report["pages"]) < len(document):
                report["flags"].append("ocr_pages_not_processed")
            report["processing_complete"] = (len(report["pages"]) == len(document)
                and all(page["status"] in {"native_text_present_no_ocr", "ocr_text_unverified"}
                        and not page["flags"] for page in report["pages"]))
            if report["ocr_calls"]:
                report["flags"].extend(["ocr_text_unverified", "ocr_confidence_unavailable", "ocr_layout_unverified"])
    except Exception as exc:
        report["flags"].append("ocr_document_error:" + type(exc).__name__)
    finally:
        report["elapsed_s"] = time.monotonic() - started
        if digest(path) != report["source_sha256"]:
            report["pages"] = []
            report["processing_complete"] = False
            report["flags"].append("ocr_source_changed_results_discarded")
        _write(output, report)
    return report


def ocr_pdf(path, *, limits=None, language="en-US"):
    limits = limits or OcrLimits()
    if not isinstance(limits, OcrLimits) or not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,2}", language):
        raise ValueError("Invalid OCR configuration")
    source = Path(path).resolve(strict=True)
    if not source.is_file() or source.suffix.casefold() != ".pdf":
        raise ValueError("OCR input must be a PDF file")
    report = _blank(source, limits, language)
    if source.stat().st_size > limits.max_file_bytes:
        report["flags"].append("ocr_file_size_limit")
        return report
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sisu-ocr-job-") as job_dir:
        request, output = Path(job_dir) / "request.json", Path(job_dir) / "result.json"
        _write(request, {"source": str(source), "limits": asdict(limits), "language": language})
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(request), str(output)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        timed_out = False
        try:
            process.wait(timeout=limits.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt" and process.poll() is None:
                subprocess.run([str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/taskkill.exe"),
                                "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                               timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
        if output.is_file():
            report = json.loads(output.read_text(encoding="utf-8"))
        if timed_out:
            report["processing_complete"] = False
            report["flags"].append("ocr_outer_timeout_partial_results")
        elif process.returncode:
            report["processing_complete"] = False
            report["flags"].append("ocr_worker_failed")
    report["wall_elapsed_s"] = time.monotonic() - started
    if report["source_sha256"] is not None:
        try:
            unchanged = digest(source) == report["source_sha256"]
        except OSError:
            unchanged = False
        if not unchanged:
            report["pages"] = []
            report["processing_complete"] = False
            report["flags"].append("ocr_source_changed_results_discarded")
    report["coverage"] = {"total_pages": report["total_pages"], "inspected_pages": len(report["pages"]),
        "native_text_pages": sum(page["status"] == "native_text_present_no_ocr" for page in report["pages"]),
        "pages_with_ocr_text": sum(page["status"] == "ocr_text_unverified" for page in report["pages"]),
        "unprocessed_or_unresolved_pages": (report["total_pages"] - sum(page["status"] in {"native_text_present_no_ocr", "ocr_text_unverified"}
                                             for page in report["pages"])) if report["total_pages"] is not None else None}
    return report


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        _extract_worker(Path(request["source"]), OcrLimits(**request["limits"]), request["language"], Path(sys.argv[3]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    report = ocr_pdf(args.pdf, limits=OcrLimits(max_pages=args.max_pages, timeout_s=args.timeout, dpi=args.dpi), language=args.language)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, report)
    print(json.dumps({key: report[key] for key in ("coverage", "ocr_calls", "wall_elapsed_s", "flags")}, indent=2))


if __name__ == "__main__":
    main()
