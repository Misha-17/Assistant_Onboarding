"""Deterministic, structure-preserving local document ingestion.

This module deliberately performs no semantic retrieval and calls no model.  It
turns supported files into immutable document revisions, real structural
sections, and source blocks that retain source order and citation locators.
Natural paragraphs, list items, headings, and table rows stay intact; only a
single pathologically large source block is split.
"""

from __future__ import annotations

import hashlib
import codecs
import math
import os
import re
import unicodedata
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from docx import Document as WordDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from .models import DocumentManifest, DocumentRevision, Section, SourceBlock


SUPPORTED_SUFFIXES = frozenset({".docx", ".pdf", ".txt", ".md", ".html", ".htm"})
_INGESTION_VERSION = "6-html-definitions-pre"

_W_P = qn("w:p")
_W_TBL = qn("w:tbl")
_W_T = qn("w:t")
_W_TAB = qn("w:tab")
_W_BR = qn("w:br")
_W_TRPR = qn("w:trPr")
_W_TBLHEADER = qn("w:tblHeader")

_WORD = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\w\"'(\[])|\n{2,}")
_MARKDOWN_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PERCENT = re.compile(r"(?<!\w)([+-]?\d+(?:[.,]\d+)?)\s*%(?!\w)")
_CURRENCY = re.compile(
    r"(?<!\w)(?:(EUR|USD|GBP|SEK|NOK|DKK|CHF|[\u20ac$\u00a3])"
    r"\s*([+-]?\d+(?:[ \u00a0.,]\d+)*)"
    r"|([+-]?\d+(?:[ \u00a0.,]\d+)*)\s*"
    r"(EUR|USD|GBP|SEK|NOK|DKK|CHF|[\u20ac$\u00a3]))(?!\w)",
    flags=re.IGNORECASE,
)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_DMY_DATE = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)")
_STRUCTURED_ID = re.compile(
    r"(?<![\w])(?=[A-Za-z0-9._/-]*[A-Za-z])(?=[A-Za-z0-9._/-]*\d)"
    r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)+(?![\w])"
)
_PHONE = re.compile(r"(?<!\w)\+?\d(?:[\s().-]*\d){6,14}(?!\w)")
_NUMBER = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?![\w])")
_ACRONYM = re.compile(
    r"(?<![\w])(?:[A-Z\u00c0-\u00de][A-Z0-9\u00c0-\u00de&-]{1,}"
    r"(?:\.[A-Z0-9\u00c0-\u00de]+)*)(?![\w])"
)
_PROPER_PHRASE = re.compile(
    r"(?<![\w])(?:[A-Z\u00c0-\u00de][\w\u00c0-\u024f'\u2019-]{1,})"
    r"(?:\s+(?:[A-Z\u00c0-\u00de][\w\u00c0-\u024f'\u2019-]{1,})){1,5}"
)


@dataclass(frozen=True, slots=True)
class ExactTerm:
    term_normalized: str
    term_kind: str
    surface: str
    document_revision_id: str
    block_id: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    document: DocumentRevision
    sections: tuple[Section, ...]
    blocks: tuple[SourceBlock, ...]
    manifest: DocumentManifest
    exact_terms: tuple[ExactTerm, ...]


@dataclass(frozen=True, slots=True)
class _SectionDraft:
    key: str
    parent_key: str | None
    depth: int
    heading: str
    section_path: str
    locator: str


@dataclass(frozen=True, slots=True)
class _BlockDraft:
    key: str
    locator: str
    kind: str
    text: str
    section_key: str = "root"
    section_path: str = ""
    table_key: str | None = None
    row_key: str | None = None
    headers: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Parsed:
    title: str
    sections: tuple[_SectionDraft, ...]
    blocks: tuple[_BlockDraft, ...]
    warnings: tuple[str, ...] = ()


def collect_files(sources: str | Path | Iterable[str | Path]) -> tuple[Path, ...]:
    """Collect supported files without crossing linked directory boundaries."""

    roots = (
        (Path(sources),)
        if isinstance(sources, (str, Path))
        else tuple(Path(item) for item in sources)
    )
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        resolved_root = root.resolve()
        candidates: Iterable[Path]
        if resolved_root.is_file():
            candidates = (resolved_root,)
        else:
            candidates = _contained_directory_files(resolved_root)
        for candidate in candidates:
            if candidate.name.startswith("~$"):
                continue
            if candidate.suffix.casefold() not in SUPPORTED_SUFFIXES:
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if resolved_root.is_dir() and not _is_relative_to(resolved, resolved_root):
                continue
            if resolved.is_file():
                found[_path_key(resolved)] = resolved
    return tuple(found[key] for key in sorted(found))


def _contained_directory_files(root: Path) -> Iterable[Path]:
    visited = {os.path.normcase(str(root))}
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        try:
            resolved_current = current_path.resolve(strict=True)
        except (OSError, RuntimeError):
            directories[:] = []
            continue
        if not _is_relative_to(resolved_current, root):
            directories[:] = []
            continue

        kept: list[str] = []
        for name in sorted(directories, key=str.casefold):
            child = current_path / name
            if _is_linked_directory(child):
                continue
            try:
                resolved_child = child.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            key = os.path.normcase(str(resolved_child))
            if not _is_relative_to(resolved_child, root) or key in visited:
                continue
            visited.add(key)
            kept.append(name)
        directories[:] = kept
        for filename in sorted(filenames, key=str.casefold):
            yield current_path / filename


def _is_linked_directory(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        # FILE_ATTRIBUTE_REPARSE_POINT; protects older pathlib on Windows too.
        return bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
    except OSError:
        return True


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def source_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_corpus(
    sources: str | Path | Iterable[str | Path],
    *,
    hard_block_tokens: int = 4096,
) -> tuple[ExtractedDocument, ...]:
    """Extract every selected file as a distinct logical document."""

    if not 512 <= int(hard_block_tokens) <= 32768:
        raise ValueError("hard_block_tokens must be between 512 and 32768")
    return tuple(
        extract_file(path, hard_block_tokens=int(hard_block_tokens))
        for path in collect_files(sources)
    )


def extract_file(path: str | Path, *, hard_block_tokens: int = 4096) -> ExtractedDocument:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported document type: {source.suffix or '<none>'}")
    if not 512 <= int(hard_block_tokens) <= 32768:
        raise ValueError("hard_block_tokens must be between 512 and 32768")

    sha256_before = source_sha256(source)

    if suffix == ".docx":
        parsed = _extract_docx(source)
    elif suffix == ".pdf":
        parsed = _extract_pdf(source)
    elif suffix == ".md":
        raw, warnings = _read_text(source)
        parsed = _extract_markdown(raw, source.stem, warnings)
    elif suffix in {".html", ".htm"}:
        raw, warnings = _read_text(source)
        parsed = _extract_html(raw, source.stem, warnings)
    else:
        raw, warnings = _read_text(source)
        parsed = _extract_plain_text(raw, source.stem, warnings)

    sha256 = source_sha256(source)
    if sha256 != sha256_before:
        raise RuntimeError(f"source changed while it was being extracted: {source}")
    logical_key = _path_key(source)
    logical_document_id = f"doc_{_digest(logical_key)[:24]}"
    # A revision identifies the extracted representation, not merely the file
    # bytes.  Parser changes and the one user-controlled structural split
    # boundary must therefore produce a new immutable revision.
    revision_digest = _digest(
        logical_document_id,
        sha256,
        _INGESTION_VERSION,
        str(int(hard_block_tokens)),
        # OCR results may vary with the installed local recognizer. Bind the
        # actual extracted representation, not only PDF bytes, into its revision.
        hashlib.sha256(repr([(b.text, b.locator, b.flags) for b in parsed.blocks]).encode("utf-8")).hexdigest()
        if any(b.kind == "pdf_ocr" for b in parsed.blocks) else "",
    )
    document_revision_id = f"rev_{revision_digest[:32]}"
    return _assemble_document(
        source=source,
        suffix=suffix,
        sha256=sha256,
        logical_document_id=logical_document_id,
        document_revision_id=document_revision_id,
        parsed=parsed,
        hard_block_tokens=int(hard_block_tokens),
    )


def _root_section(title: str) -> _SectionDraft:
    return _SectionDraft(
        key="root",
        parent_key=None,
        depth=0,
        heading=title,
        section_path=title,
        locator="document",
    )


def _extract_docx(path: Path) -> _Parsed:
    word = WordDocument(path)
    title = path.stem
    sections: list[_SectionDraft] = [_root_section(title)]
    blocks: list[_BlockDraft] = []
    warnings: list[str] = []
    heading_stack: list[_SectionDraft] = []
    paragraph_number = 0
    table_number = 0

    drawing_count = len(word.element.body.xpath(".//w:drawing"))
    if drawing_count:
        warnings.append(f"docx_unindexed_drawings:{drawing_count}")

    for child in word.element.body.iterchildren():
        if child.tag == _W_P:
            paragraph_number += 1
            paragraph = Paragraph(child, word)
            text = _source_text(_xml_text(child))
            if not text:
                continue
            level = _word_heading_level(paragraph)
            if level is not None:
                while heading_stack and heading_stack[-1].depth >= level:
                    heading_stack.pop()
                parent = heading_stack[-1] if heading_stack else sections[0]
                section = _SectionDraft(
                    key=f"docx:heading:{paragraph_number}",
                    parent_key=parent.key,
                    depth=level,
                    heading=text,
                    section_path=" > ".join((*_section_titles(heading_stack), text)),
                    locator=f"paragraph {paragraph_number}",
                )
                sections.append(section)
                heading_stack.append(section)
                blocks.append(_BlockDraft(
                    key=f"paragraph:{paragraph_number}:heading",
                    locator=f"paragraph {paragraph_number}",
                    kind="section_header",
                    text=text,
                    section_key=section.key,
                    section_path=section.section_path,
                ))
                continue

            current = heading_stack[-1] if heading_stack else sections[0]
            blocks.append(_BlockDraft(
                key=f"paragraph:{paragraph_number}",
                locator=f"paragraph {paragraph_number}",
                kind="list_item" if _word_is_list(paragraph) else "prose",
                text=text,
                section_key=current.key,
                section_path=current.section_path,
            ))
        elif child.tag == _W_TBL:
            table_number += 1
            current = heading_stack[-1] if heading_stack else sections[0]
            blocks.extend(_docx_table_blocks(
                Table(child, word),
                table_number=table_number,
                section=current,
            ))

    footnotes = _docx_footnotes(path)
    if footnotes:
        section = _SectionDraft(
            key="docx:footnotes",
            parent_key="root",
            depth=1,
            heading="Footnotes",
            section_path="Footnotes",
            locator="footnotes",
        )
        sections.append(section)
        blocks.extend(replace(item, section_key=section.key, section_path=section.section_path) for item in footnotes)

    return _Parsed(title, tuple(sections), tuple(blocks), tuple(_unique(warnings)))


def _xml_text(element: Any) -> str:
    pieces: list[str] = []
    for node in element.iter():
        if node.tag == _W_T:
            pieces.append(node.text or "")
        elif node.tag == _W_TAB:
            pieces.append("\t")
        elif node.tag == _W_BR:
            pieces.append("\n")
    return "".join(pieces)


def _word_heading_level(paragraph: Paragraph) -> int | None:
    try:
        style = paragraph.style.name or ""
    except (AttributeError, KeyError):
        return None
    match = re.match(r"Heading\s+([1-9])\b", style, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1 if style.casefold() in {"title", "subtitle"} else None


def _word_is_list(paragraph: Paragraph) -> bool:
    try:
        style = (paragraph.style.name or "").casefold()
    except (AttributeError, KeyError):
        style = ""
    if "list" in style:
        return True
    properties = paragraph._p.pPr
    return bool(properties is not None and properties.numPr is not None)


def _docx_table_blocks(
    table: Table,
    *,
    table_number: int,
    section: _SectionDraft,
) -> list[_BlockDraft]:
    rows: list[tuple[int, tuple[str, ...], bool]] = []
    for row_number, row in enumerate(table.rows, start=1):
        values_list: list[str] = []
        previous_cell_xml: Any | None = None
        for cell in row.cells:
            # python-docx repeats the same underlying cell for a horizontal
            # merge. Retain its grid position as blank rather than duplicating
            # the merged title or label several times.
            if cell._tc is previous_cell_xml:
                values_list.append("")
            else:
                values_list.append(_source_text(cell.text))
            previous_cell_xml = cell._tc
        values = tuple(values_list)
        if not any(values):
            continue
        tr_properties = row._tr.find(_W_TRPR)
        explicit_header = bool(
            tr_properties is not None
            and tr_properties.find(_W_TBLHEADER) is not None
        )
        rows.append((row_number, values, explicit_header))
    if not rows:
        return []

    explicit = [item for item in rows if item[2]]
    header_row = explicit[0] if explicit else _inferred_docx_header(rows)
    header_number = header_row[0]
    headers = _header_cells(header_row[1])
    table_key = f"docx:table:{table_number}"
    inferred_flags = () if explicit else ("table_header_inferred",)
    blocks: list[_BlockDraft] = []
    for row_number, values, _ in rows:
        if row_number == header_number:
            blocks.append(_BlockDraft(
                key=f"{table_key}:header",
                locator=f"table {table_number}, row {row_number} header",
                kind="table_header",
                text=" | ".join(headers),
                section_key=section.key,
                section_path=section.section_path,
                table_key=table_key,
                headers=headers,
                flags=inferred_flags,
            ))
            continue
        row_flags = inferred_flags
        if row_number < header_number:
            row_flags = (*row_flags, "table_preamble")
        blocks.append(_BlockDraft(
            key=f"{table_key}:row:{row_number}",
            locator=f"table {table_number}, row {row_number}",
            kind="table_row",
            text=" | ".join(values),
            section_key=section.key,
            section_path=section.section_path,
            table_key=table_key,
            row_key=f"{table_key}:row:{row_number}",
            headers=headers,
            flags=row_flags,
        ))
    return blocks


def _inferred_docx_header(
    rows: Sequence[tuple[int, tuple[str, ...], bool]],
) -> tuple[int, tuple[str, ...], bool]:
    """Avoid treating a merged table title as the column header.

    Word often represents a merged title cell by repeating the same value in
    every grid position.  When that happens, the first early row containing
    multiple distinct labels is a materially better structural header.  No
    row is discarded: title rows remain ordered table-row evidence.
    """

    first = rows[0]
    first_distinct = {
        normalize_surface(value) for value in first[1] if normalize_surface(value)
    }
    if len(first_distinct) > 1 or len(first[1]) <= 1:
        return first
    for candidate in rows[1:6]:
        distinct = {
            normalize_surface(value)
            for value in candidate[1]
            if normalize_surface(value)
        }
        if len(distinct) > 1:
            return candidate
    return first


def _docx_footnotes(path: Path) -> list[_BlockDraft]:
    try:
        with zipfile.ZipFile(path) as package:
            if "word/footnotes.xml" not in package.namelist():
                return []
            root = ElementTree.fromstring(package.read("word/footnotes.xml"))
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError):
        return []

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    blocks: list[_BlockDraft] = []
    for note in root.findall(f"{namespace}footnote"):
        identifier = note.get(f"{namespace}id", "")
        if identifier.startswith("-"):
            continue
        text = _source_text(" ".join(node.text or "" for node in note.iter(f"{namespace}t")))
        if text:
            blocks.append(_BlockDraft(
                key=f"footnote:{identifier}",
                locator=f"footnote {identifier}",
                kind="footnote",
                text=text,
            ))
    return blocks


def _extract_pdf(path: Path) -> _Parsed:
    reader = PdfReader(path, strict=False)
    title = path.stem
    sections: list[_SectionDraft] = [_root_section(title)]
    blocks: list[_BlockDraft] = []
    warnings: list[str] = []
    if getattr(reader, "is_encrypted", False):
        warnings.append("pdf_encrypted_but_readable")

    table_number = 0
    ocr_report = None
    ocr_enabled = os.environ.get("SISU_READER_PDF_OCR", "off").casefold() == "windows"
    for page_number, page in enumerate(reader.pages, start=1):
        section = _SectionDraft(
            key=f"pdf:page:{page_number}",
            parent_key="root",
            depth=1,
            heading=f"Page {page_number}",
            section_path=f"Page {page_number}",
            locator=f"page {page_number}",
        )
        sections.append(section)
        try:
            plain = page.extract_text() or ""
        except Exception as exc:
            warnings.append(f"pdf_page_extract_error:page={page_number}:{type(exc).__name__}")
            plain = ""
        try:
            layout = page.extract_text(extraction_mode="layout") or ""
        except (TypeError, ValueError, NotImplementedError):
            layout = ""
        except Exception:
            layout = ""

        source = layout or plain
        if not _source_text(source):
            if ocr_enabled:
                try:
                    if ocr_report is None:
                        from .local_pdf_ocr import ocr_pdf
                        ocr_report = ocr_pdf(path, language=os.environ.get("SISU_READER_OCR_LANGUAGE", "en-US"))
                        warnings.extend("pdf_ocr:" + flag for flag in ocr_report["flags"])
                    ocr_page = next((p for p in ocr_report["pages"] if p["page"] == page_number), None)
                    if ocr_page and ocr_page["status"] == "ocr_text_unverified" and ocr_page.get("text", "").strip():
                        regions = ocr_page["regions"]
                        box = [min(r["box_pdf_points"][0] for r in regions), min(r["box_pdf_points"][1] for r in regions),
                               max(r["box_pdf_points"][2] for r in regions), max(r["box_pdf_points"][3] for r in regions)]
                        blocks.append(_BlockDraft(key=f"pdf:page:{page_number}:ocr",
                            locator=f"page {page_number}, OCR transcription unverified, bbox=" + ",".join(map(str, box)),
                            kind="pdf_ocr", text=ocr_page["text"], section_key=section.key,
                            section_path=section.section_path,
                            flags=("ocr_text_unverified", "ocr_confidence_unavailable", "ocr_layout_unverified",
                                   "ocr_backend_version:" + str(ocr_page.get("backend_version", "unknown")))))
                        warnings.append(f"pdf_ocr_uncertain:page={page_number}")
                        continue
                except Exception as exc:
                    warnings.append(f"pdf_ocr_error:page={page_number}:{type(exc).__name__}")
            warnings.append(f"pdf_needs_ocr:page={page_number}")
            continue
        lines = source.splitlines()
        groups, excluded = _layout_table_groups(lines)
        for group in groups:
            table_number += 1
            header_number, header_cells = group[0]
            blocks.extend(_table_blocks(
                headers=header_cells,
                data=((number, cells) for number, cells in group[1:]),
                table_key=f"pdf:table:{table_number}",
                header_key=f"pdf:table:{table_number}:header",
                header_locator=(
                    f"page {page_number}, line {header_number}, table {table_number} header"
                ),
                row_locator=lambda number, p=page_number, t=table_number: (
                    f"page {p}, line {number}, table {t}"
                ),
                section=section,
                header_flags=("pdf_layout_table_inferred",),
            ))

        prose_lines = [
            line for number, line in enumerate(lines, start=1) if number not in excluded
        ]
        prose = _source_text("\n".join(prose_lines))
        if not prose and plain and layout:
            prose = _source_text(plain)
        for part, value in enumerate(_paragraphs(prose), start=1):
            blocks.append(_BlockDraft(
                key=f"pdf:page:{page_number}:text:{part}",
                locator=f"page {page_number}, text {part}",
                kind="pdf_text",
                text=value,
                section_key=section.key,
                section_path=section.section_path,
            ))
    return _Parsed(title, tuple(sections), tuple(blocks), tuple(_unique(warnings)))


def _layout_table_groups(
    lines: Sequence[str],
) -> tuple[list[list[tuple[int, tuple[str, ...]]]], set[int]]:
    candidates: list[tuple[int, tuple[str, ...]] | None] = []
    for number, line in enumerate(lines, start=1):
        cells = tuple(
            _source_text(value)
            for value in re.split(r"\s{2,}", line.strip())
            if _source_text(value)
        )
        candidates.append((number, cells) if len(cells) >= 2 else None)

    raw_groups: list[list[tuple[int, tuple[str, ...]]]] = []
    pending: list[tuple[int, tuple[str, ...]]] = []
    for item in candidates:
        if item is None:
            if pending:
                raw_groups.append(pending)
                pending = []
        else:
            pending.append(item)
    if pending:
        raw_groups.append(pending)

    groups: list[list[tuple[int, tuple[str, ...]]]] = []
    excluded: set[int] = set()
    for group in raw_groups:
        widths = [len(cells) for _, cells in group]
        data_has_number = any(re.search(r"\d", cell) for _, cells in group[1:] for cell in cells)
        if len(group) < 3 or max(widths) - min(widths) > 1 or not data_has_number:
            continue
        groups.append(group)
        excluded.update(number for number, _ in group)
    return groups, excluded


def _read_text(path: Path) -> tuple[str, tuple[str, ...]]:
    raw = path.read_bytes()
    # UTF-32 LE shares UTF-16 LE's prefix, so test the longer BOM first.
    if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        text, warnings = raw.decode("utf-32"), ("decoded_as_utf32",)
    elif raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text, warnings = raw.decode("utf-16"), ("decoded_as_utf16",)
    else:
        try:
            text, warnings = raw.decode("utf-8-sig"), ()
        except UnicodeDecodeError:
            text, warnings = raw.decode("cp1252"), ("decoded_as_cp1252",)
    if "\x00" in text:
        raise ValueError("Text contains NUL characters; use a valid UTF-8 file or a UTF-16/UTF-32 file with a byte-order mark")
    return text, warnings


def _extract_plain_text(text: str, title: str, warnings: Sequence[str]) -> _Parsed:
    root = _root_section(title)
    blocks = tuple(
        _BlockDraft(
            key=f"paragraph:{number}",
            locator=f"paragraph {number}",
            kind="prose",
            text=value,
            section_path=root.section_path,
        )
        for number, value in enumerate(_paragraphs(text), start=1)
    )
    return _Parsed(title, (root,), blocks, tuple(warnings))


def _extract_markdown(text: str, title: str, warnings: Sequence[str]) -> _Parsed:
    lines = text.splitlines()
    root = _root_section(title)
    sections: list[_SectionDraft] = [root]
    blocks: list[_BlockDraft] = []
    headings: list[_SectionDraft] = []
    paragraph: list[str] = []
    paragraph_start = 1
    table_number = 0

    def current_section() -> _SectionDraft:
        return headings[-1] if headings else root

    def flush(end_line: int) -> None:
        nonlocal paragraph
        value = _source_text("\n".join(paragraph))
        if value:
            current = current_section()
            is_list = all(
                re.match(r"^\s*(?:[-*+] |\d+[.)] )", line) for line in paragraph
            )
            if is_list:
                for offset, line in enumerate(paragraph):
                    line_number = paragraph_start + offset
                    blocks.append(_BlockDraft(
                        key=f"markdown:line:{line_number}:list",
                        locator=f"line {line_number}",
                        kind="list_item",
                        text=_source_text(line),
                        section_key=current.key,
                        section_path=current.section_path,
                    ))
            else:
                blocks.append(_BlockDraft(
                    key=f"markdown:lines:{paragraph_start}-{end_line}",
                    locator=f"lines {paragraph_start}-{end_line}",
                    kind="prose",
                    text=value,
                    section_key=current.key,
                    section_path=current.section_path,
                ))
        paragraph = []

    index = 0
    while index < len(lines):
        line = lines[index]
        setext = (
            re.fullmatch(r"\s*(=+|-+)\s*", lines[index + 1])
            if line.strip() and index + 1 < len(lines)
            else None
        )
        if setext and len(setext.group(1)) >= 3:
            flush(index)
            level = 1 if setext.group(1).startswith("=") else 2
            value = _source_text(line)
            while headings and headings[-1].depth >= level:
                headings.pop()
            parent = headings[-1] if headings else root
            section = _SectionDraft(
                key=f"markdown:line:{index + 1}:setext-heading",
                parent_key=parent.key,
                depth=level,
                heading=value,
                section_path=" > ".join((*_section_titles(headings), value)),
                locator=f"lines {index + 1}-{index + 2}",
            )
            sections.append(section)
            headings.append(section)
            blocks.append(_BlockDraft(
                key=section.key,
                locator=section.locator,
                kind="section_header",
                text=value,
                section_key=section.key,
                section_path=section.section_path,
            ))
            index += 2
            paragraph_start = index + 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush(index)
            level = len(heading.group(1))
            value = _source_text(heading.group(2))
            while headings and headings[-1].depth >= level:
                headings.pop()
            parent = headings[-1] if headings else root
            section = _SectionDraft(
                key=f"markdown:line:{index + 1}:heading",
                parent_key=parent.key,
                depth=level,
                heading=value,
                section_path=" > ".join((*_section_titles(headings), value)),
                locator=f"line {index + 1}",
            )
            sections.append(section)
            headings.append(section)
            blocks.append(_BlockDraft(
                key=section.key,
                locator=section.locator,
                kind="section_header",
                text=value,
                section_key=section.key,
                section_path=section.section_path,
            ))
            index += 1
            paragraph_start = index + 1
            continue

        if index + 1 < len(lines) and "|" in line and _MARKDOWN_TABLE_RULE.match(lines[index + 1]):
            flush(index)
            table_number += 1
            data: list[tuple[int, tuple[str, ...]]] = []
            cursor = index + 2
            while cursor < len(lines) and "|" in lines[cursor] and lines[cursor].strip():
                data.append((cursor + 1, tuple(_markdown_cells(lines[cursor]))))
                cursor += 1
            current = current_section()
            blocks.extend(_table_blocks(
                headers=_markdown_cells(line),
                data=data,
                table_key=f"markdown:table:{table_number}",
                header_key=f"markdown:table:{table_number}:header",
                header_locator=f"line {index + 1}, table {table_number} header",
                row_locator=lambda number, t=table_number: f"line {number}, table {t}",
                section=current,
            ))
            index = cursor
            paragraph_start = index + 1
            continue

        if not line.strip():
            flush(index)
            index += 1
            paragraph_start = index + 1
            continue
        if not paragraph:
            paragraph_start = index + 1
        paragraph.append(line)
        index += 1
    flush(len(lines))
    return _Parsed(title, tuple(sections), tuple(blocks), tuple(warnings))


def _markdown_cells(line: str) -> list[str]:
    return [_source_text(cell) for cell in line.strip().strip("|").split("|")]


def _extract_html(text: str, fallback_title: str, warnings: Sequence[str]) -> _Parsed:
    soup = BeautifulSoup(text, "html.parser")
    for unwanted in soup(["script", "style", "noscript", "template"]):
        unwanted.decompose()
    html_title = _source_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    title = html_title or fallback_title
    root_section = _root_section(title)
    sections: list[_SectionDraft] = [root_section]
    blocks: list[_BlockDraft] = []
    headings: list[_SectionDraft] = []
    root = soup.body or soup
    element_number = 0
    table_number = 0
    extraction_warnings = list(warnings)

    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "dl", "pre"]):
        if not isinstance(element, Tag):
            continue
        # Each selected container owns its descendants. In particular, a dd's
        # paragraphs and nested dl are emitted by its definition group once.
        if element.find_parent(["table", "dl", "pre", "p", "li"]) is not None:
            continue
        element_number += 1
        current = headings[-1] if headings else root_section

        if element.name == "dl":
            for group_number, (terms, value, flags) in enumerate(_html_definition_groups(element), 1):
                if not value.strip():
                    continue
                extraction_warnings.extend(flag for flag in flags if flag.startswith("html_definition_unassociated"))
                blocks.append(_BlockDraft(
                    key=f"html:element:{element_number}:definition:{group_number}",
                    locator=f"element {element_number}, definition group {group_number}",
                    kind="prose", text=value,
                    section_key=current.key, section_path=current.section_path,
                    headers=terms, flags=flags,
                ))
            continue

        if element.name == "pre":
            value = _preformatted_text(element.get_text())
            if value.strip():
                blocks.append(_BlockDraft(
                    key=f"html:element:{element_number}", locator=f"element {element_number}, preformatted",
                    kind="prose", text=value, section_key=current.key,
                    section_path=current.section_path, flags=("html_preformatted",),
                ))
            continue

        if element.name == "table":
            table_number += 1
            rows: list[tuple[int, tuple[str, ...], bool]] = []
            table_has_pre = element.find("pre") is not None
            for row_number, row in enumerate(element.find_all("tr"), start=1):
                if row.find_parent("table") is not element:
                    continue
                cells = row.find_all(["th", "td"], recursive=False)
                values = tuple(_html_visible_text(cell)[0] for cell in cells)
                if any(values):
                    rows.append((row_number, values, any(cell.name == "th" for cell in cells)))
            if rows:
                header = next((item for item in rows if item[2]), rows[0])
                blocks.extend(_table_blocks(
                    headers=header[1],
                    data=((number, values) for number, values, _ in rows if number != header[0]),
                    table_key=f"html:table:{table_number}",
                    header_key=f"html:table:{table_number}:header",
                    header_locator=f"table {table_number}, row {header[0]} header",
                    row_locator=lambda number, t=table_number: f"table {t}, row {number}",
                    section=current,
                    header_flags=(
                        *(() if header[2] else ("table_header_inferred",)),
                        *(("html_preformatted",) if table_has_pre else ()),
                    ),
                ))
            continue

        value, contains_pre = _html_visible_text(element)
        if not value:
            continue
        if element.name and element.name.startswith("h"):
            level = int(element.name[1])
            while headings and headings[-1].depth >= level:
                headings.pop()
            parent = headings[-1] if headings else root_section
            section = _SectionDraft(
                key=f"html:element:{element_number}:heading",
                parent_key=parent.key,
                depth=level,
                heading=value,
                section_path=" > ".join((*_section_titles(headings), value)),
                locator=f"element {element_number}",
            )
            sections.append(section)
            headings.append(section)
            blocks.append(_BlockDraft(
                key=section.key,
                locator=section.locator,
                kind="section_header",
                text=value,
                section_key=section.key,
                section_path=section.section_path,
            ))
        else:
            current = headings[-1] if headings else root_section
            blocks.append(_BlockDraft(
                key=f"html:element:{element_number}",
                locator=f"element {element_number}",
                kind="list_item" if element.name == "li" else "prose",
                text=value,
                section_key=current.key,
                section_path=current.section_path,
                flags=("html_preformatted",) if contains_pre else (),
            ))
    return _Parsed(title, tuple(sections), tuple(blocks), tuple(extraction_warnings))


def _preformatted_text(text: str) -> str:
    """Normalize transport line endings, retaining meaningful code whitespace."""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _html_visible_text(element: Tag) -> tuple[str, bool]:
    """Render visible block boundaries once, leaving preformatted regions intact.

    The result is canonical extracted text, not a claim of byte-identical HTML.
    Script/style/template nodes have already been removed by the caller.
    """
    sections: list[str] = []
    ordinary: list[str] = []
    contains_pre = False
    boundaries = {"p", "li", "dt", "dd", "dl", "div", "section", "article", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def flush() -> None:
        value = _source_text("".join(ordinary))
        ordinary.clear()
        if value:
            sections.append(value)

    def visit(node: Any) -> None:
        nonlocal contains_pre
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            ordinary.append(str(node))
            return
        if not isinstance(node, Tag):
            return
        if node.name == "pre":
            flush()
            value = _preformatted_text(node.get_text())
            if value.strip():
                sections.append(value)
            contains_pre = True
            return
        if node.name == "br":
            ordinary.append("\n")
            return
        if node.name in boundaries:
            ordinary.append("\n")
        for child in node.children:
            visit(child)
        if node is not element and node.name in {"td", "th"} and node.find_next_sibling(["td", "th"]) is not None:
            ordinary.append(" | ")
        if node.name in boundaries:
            ordinary.append("\n")

    visit(element)
    flush()
    return "\n".join(sections), contains_pre


def _html_definition_groups(element: Tag) -> Iterable[tuple[tuple[str, ...], str, tuple[str, ...]]]:
    """Bind each consecutive dt group to its dd group in source order.

    HTML5 div wrappers are transparent. Nested lists remain within their own
    outer description instead of being emitted again or borrowing a sibling's
    label. Orphan terms/descriptions are retained and explicitly flagged.
    """
    terms: list[str] = []
    descriptions: list[str] = []
    has_description = False
    has_pre = False

    def entries(parent: Tag) -> Iterable[Any]:
        for child in parent.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, Tag) and child.name == "div":
                yield from entries(child)
            elif isinstance(child, Tag) or (isinstance(child, NavigableString) and str(child).strip()):
                yield child

    def group() -> tuple[tuple[str, ...], str, tuple[str, ...]]:
        flags = ["html_definition_group"]
        if not terms or not has_description:
            flags.append("html_definition_unassociated_term_or_description")
        if has_pre:
            flags.append("html_preformatted")
        return tuple(terms), "\n".join((*terms, *descriptions)), tuple(flags)

    for entry in entries(element):
        name = entry.name if isinstance(entry, Tag) else None
        if name == "dt":
            if has_description:
                yield group()
                terms, descriptions, has_description, has_pre = [], [], False, False
            value, pre = _html_visible_text(entry)
            if value.strip():
                terms.append(value)
            has_pre |= pre
        elif name == "dd":
            value, pre = _html_visible_text(entry)
            if value.strip():
                descriptions.append(value)
            has_description = True
            has_pre |= pre
        else:
            if terms or has_description:
                yield group()
                terms, descriptions, has_description, has_pre = [], [], False, False
            value, pre = _html_visible_text(entry) if isinstance(entry, Tag) else (_source_text(str(entry)), False)
            if value:
                yield (), value, ("html_definition_unassociated_content", *(("html_preformatted",) if pre else ()))
    if terms or has_description:
        yield group()


def _table_blocks(
    *,
    headers: Sequence[str],
    data: Iterable[tuple[int, Sequence[str]]],
    table_key: str,
    header_key: str,
    header_locator: str,
    row_locator: Any,
    section: _SectionDraft,
    header_flags: tuple[str, ...] = (),
) -> list[_BlockDraft]:
    clean_headers = _header_cells(headers)
    blocks = [_BlockDraft(
        key=header_key,
        locator=header_locator,
        kind="table_header",
        text=" | ".join(clean_headers),
        section_key=section.key,
        section_path=section.section_path,
        table_key=table_key,
        headers=clean_headers,
        flags=header_flags,
    )]
    for row_number, row in data:
        cells = tuple((_preformatted_text(str(value)) if "html_preformatted" in header_flags else _source_text(str(value))) for value in row)
        if not any(cells):
            continue
        blocks.append(_BlockDraft(
            key=f"{table_key}:row:{row_number}",
            locator=str(row_locator(row_number)),
            kind="table_row",
            text=" | ".join(cells),
            section_key=section.key,
            section_path=section.section_path,
            table_key=table_key,
            row_key=f"{table_key}:row:{row_number}",
            headers=clean_headers,
            flags=header_flags,
        ))
    return blocks


def _assemble_document(
    *,
    source: Path,
    suffix: str,
    sha256: str,
    logical_document_id: str,
    document_revision_id: str,
    parsed: _Parsed,
    hard_block_tokens: int,
) -> ExtractedDocument:
    section_drafts = list(parsed.sections) or [_root_section(parsed.title)]
    if section_drafts[0].key != "root":
        section_drafts.insert(0, _root_section(parsed.title))
    section_ids = {
        item.key: f"sec_{_digest(document_revision_id, item.key)[:24]}"
        for item in section_drafts
    }

    expanded: list[_BlockDraft] = []
    for draft in parsed.blocks:
        preserve_whitespace = "html_preformatted" in draft.flags
        text = _preformatted_text(draft.text) if preserve_whitespace else _source_text(draft.text)
        if not text.strip():
            continue
        parts = _split_preformatted(text, hard_block_tokens) if preserve_whitespace else _split_pathological(text, hard_block_tokens)
        for part_number, part in enumerate(parts, start=1):
            split = len(parts) > 1
            expanded.append(replace(
                draft,
                key=f"{draft.key}:part:{part_number}" if split else draft.key,
                locator=f"{draft.locator}, part {part_number}" if split else draft.locator,
                text=part,
                flags=tuple(dict.fromkeys((
                    *draft.flags,
                    *(("pathological_source_block_split",) if split else ()),
                ))),
            ))

    block_ids = [
        f"blk_{_digest(document_revision_id, item.key, str(index), item.locator, item.text)[:24]}"
        for index, item in enumerate(expanded)
    ]
    table_ids: dict[str, str] = {}
    row_ids: dict[str, str] = {}
    for item in expanded:
        if item.table_key:
            table_ids.setdefault(
                item.table_key,
                f"tbl_{_digest(document_revision_id, item.table_key)[:24]}",
            )
        if item.row_key:
            row_ids.setdefault(
                item.row_key,
                f"row_{_digest(document_revision_id, item.row_key)[:24]}",
            )

    offsets: list[tuple[int, int]] = []
    canonical_parts: list[str] = []
    cursor = 0
    for item in expanded:
        start = cursor
        end = start + len(item.text)
        offsets.append((start, end))
        canonical_parts.append(item.text)
        cursor = end + 2
    canonical_text = "\n\n".join(canonical_parts)
    body_sha256 = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()

    warnings = list(parsed.warnings)
    if not expanded:
        warnings.append("no_extractable_text")
        coverage = "empty"
    elif any(
        warning.startswith((
            "pdf_needs_ocr",
            "pdf_ocr",
            "pdf_page_extract_error",
            "docx_unindexed_drawings",
            "html_definition_unassociated",
        ))
        for warning in warnings
    ):
        coverage = "partial"
    else:
        coverage = "complete"

    document = DocumentRevision(
        document_revision_id=document_revision_id,
        logical_document_id=logical_document_id,
        title=parsed.title,
        source_path=str(source),
        source_sha256=sha256,
        file_type=suffix.removeprefix("."),
        extraction_coverage=coverage,
        warnings=tuple(_unique(warnings)),
        token_estimate=_token_estimate(canonical_text),
        body_sha256=body_sha256,
    )

    blocks: list[SourceBlock] = []
    for ordinal, (draft, block_id, offset) in enumerate(zip(expanded, block_ids, offsets, strict=True)):
        blocks.append(SourceBlock(
            block_id=block_id,
            document_revision_id=document_revision_id,
            section_id=section_ids.get(draft.section_key, section_ids["root"]),
            ordinal=ordinal,
            kind=draft.kind,
            locator=draft.locator,
            text=draft.text,
            text_sha256=hashlib.sha256(draft.text.encode("utf-8")).hexdigest(),
            canonical_char_start=offset[0],
            canonical_char_end=offset[1],
            previous_block_id=block_ids[ordinal - 1] if ordinal else None,
            next_block_id=block_ids[ordinal + 1] if ordinal + 1 < len(block_ids) else None,
            table_id=table_ids.get(draft.table_key or ""),
            row_id=row_ids.get(draft.row_key or ""),
            headers=draft.headers,
            token_estimate=_token_estimate(draft.text),
            extraction_flags=draft.flags,
        ))

    children: dict[str, list[str]] = {item.key: [] for item in section_drafts}
    for item in section_drafts:
        if item.parent_key in children:
            children[item.parent_key].append(item.key)

    def descendants(key: str) -> set[str]:
        result = {key}
        pending = list(children.get(key, ()))
        while pending:
            child = pending.pop()
            if child in result:
                continue
            result.add(child)
            pending.extend(children.get(child, ()))
        return result

    block_section_keys = [item.section_key for item in expanded]
    sections: list[Section] = []
    for ordinal, draft in enumerate(section_drafts):
        included = descendants(draft.key)
        positions = [
            position
            for position, section_key in enumerate(block_section_keys)
            if section_key in included
        ]
        first = positions[0] if positions else -1
        last = positions[-1] if positions else -1
        token_count = sum(blocks[position].token_estimate for position in positions)
        sections.append(Section(
            section_id=section_ids[draft.key],
            document_revision_id=document_revision_id,
            parent_section_id=section_ids.get(draft.parent_key or ""),
            ordinal=ordinal,
            depth=draft.depth,
            heading=draft.heading,
            section_path=draft.section_path,
            locator=draft.locator,
            first_block_ordinal=first,
            last_block_ordinal=last,
            token_estimate=token_count,
        ))

    exact_terms: list[ExactTerm] = []
    manifest_texts = [parsed.title, *(item.heading for item in section_drafts[1:])]
    exact_surfaces: list[str] = list(manifest_texts)
    for manifest_text in manifest_texts:
        exact_surfaces.extend(surface for _, surface in extract_exact_surfaces(manifest_text))
    seen_terms: set[tuple[str, str, str]] = set()
    for block in blocks:
        for kind, surface in extract_exact_surfaces(block.text):
            normalized = normalize_surface(surface)
            key = (normalized, kind, block.block_id)
            if not normalized or key in seen_terms:
                continue
            seen_terms.add(key)
            exact_terms.append(ExactTerm(
                term_normalized=normalized,
                term_kind=kind,
                surface=surface,
                document_revision_id=document_revision_id,
                block_id=block.block_id,
            ))
            exact_surfaces.append(surface)

    lead_parts: list[str] = []
    lead_chars = 0
    for block in blocks:
        if block.kind == "section_header":
            continue
        separator_chars = 2 if lead_parts else 0
        remaining = 4000 - lead_chars - separator_chars
        if remaining <= 0:
            break
        lead_parts.append(block.text[:remaining].rstrip())
        lead_chars += separator_chars + len(lead_parts[-1])
        if len(block.text) > remaining:
            break
        if len(lead_parts) >= 3:
            break

    outline = "\n".join(item.section_path for item in section_drafts[1:])
    manifest = DocumentManifest(
        manifest_id=f"manifest_{_digest(document_revision_id, 'manifest-v1')[:24]}",
        document_revision_id=document_revision_id,
        title=parsed.title,
        source_path=str(source),
        file_type=suffix.removeprefix("."),
        extraction_coverage=coverage,
        outline=outline,
        lead_text="\n\n".join(lead_parts),
        exact_surfaces=tuple(_unique(exact_surfaces)),
        token_estimate=document.token_estimate,
        warnings=document.warnings,
    )
    return ExtractedDocument(
        document=document,
        sections=tuple(sections),
        blocks=tuple(blocks),
        manifest=manifest,
        exact_terms=tuple(exact_terms),
    )


def extract_exact_surfaces(text: str) -> tuple[tuple[str, str], ...]:
    """Return deterministic literal/entity surfaces for exact enumeration."""

    values: list[tuple[str, str]] = []
    values.extend(("email", match.group(0)) for match in _EMAIL.finditer(text))
    values.extend(("date", match.group(0)) for match in _ISO_DATE.finditer(text))
    values.extend(("date", match.group(0)) for match in _DMY_DATE.finditer(text))
    values.extend(("percent", match.group(0)) for match in _PERCENT.finditer(text))
    values.extend(("currency", match.group(0)) for match in _CURRENCY.finditer(text))
    values.extend(("identifier", match.group(0)) for match in _STRUCTURED_ID.finditer(text))
    values.extend(("phone", match.group(0)) for match in _PHONE.finditer(text))
    values.extend(("acronym", match.group(0)) for match in _ACRONYM.finditer(text))
    values.extend(("proper_phrase", match.group(0)) for match in _PROPER_PHRASE.finditer(text))
    values.extend(("number", match.group(0)) for match in _NUMBER.finditer(text))
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, raw in values:
        surface = _source_text(raw)
        key = (kind, normalize_surface(surface))
        if surface and key[1] and key not in seen:
            result.append((kind, surface))
            seen.add(key)
    return tuple(result)


def normalize_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = normalized.replace("\u2019", "'").replace("\u2010", "-").replace("\u2011", "-")
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized


def _split_pathological(text: str, maximum_tokens: int) -> tuple[str, ...]:
    if _token_estimate(text) <= maximum_tokens:
        return (text,)
    sentences = [
        _source_text(value) for value in _SENTENCE_BOUNDARY.split(text) if _source_text(value)
    ]
    output: list[str] = []
    pending: list[str] = []
    for sentence in sentences:
        if _token_estimate(sentence) > maximum_tokens:
            if pending:
                output.append(" ".join(pending))
                pending = []
            output.extend(_split_words(sentence, maximum_tokens))
            continue
        proposed = " ".join((*pending, sentence))
        if pending and _token_estimate(proposed) > maximum_tokens:
            output.append(" ".join(pending))
            pending = []
        pending.append(sentence)
    if pending:
        output.append(" ".join(pending))
    return tuple(item for item in output if item)


def _split_words(text: str, maximum_tokens: int) -> list[str]:
    output: list[str] = []
    pending: list[str] = []
    for word in text.split():
        if _token_estimate(word) > maximum_tokens:
            if pending:
                output.append(" ".join(pending))
                pending = []
            # A base64 dump, minified payload, or corrupt OCR token can have
            # no usable boundary.  A maximum_tokens-sized character slice is
            # conservative because the estimator can never count more token
            # units than characters.
            output.extend(
                word[start:start + maximum_tokens]
                for start in range(0, len(word), maximum_tokens)
            )
            continue
        proposed = " ".join((*pending, word))
        if pending and _token_estimate(proposed) > maximum_tokens:
            output.append(" ".join(pending))
            pending = []
        pending.append(word)
    if pending:
        output.append(" ".join(pending))
    return output


def _paragraphs(text: str) -> list[str]:
    # Splitting first is deliberately simpler than a look-ahead matcher here.
    # The old expression required the final character before EOF to be
    # non-whitespace, so an entirely ordinary newline-terminated text file was
    # silently indexed as an empty document.
    values = [_source_text(value) for value in re.split(r"\n\s*\n", text)]
    return [value for value in values if value]


def _token_estimate(text: str) -> int:
    if not text:
        return 0
    return max(len(_WORD.findall(text)), math.ceil(len(text) / 4))


def _section_titles(sections: Sequence[_SectionDraft]) -> tuple[str, ...]:
    return tuple(section.heading for section in sections if section.heading)


def _header_cells(values: Iterable[str]) -> tuple[str, ...]:
    # Column position is evidence; never deduplicate or collapse blank headers.
    return tuple(_source_text(str(value)) for value in values)


def _split_preformatted(text: str, maximum_tokens: int) -> tuple[str, ...]:
    """Bound huge examples using contiguous slices, never a word re-join.

    Prefer a line boundary when a block must split; a single oversized line is
    divided without dropping any characters. Concatenating parts reconstructs
    the original canonical preformatted value exactly.
    """
    output: list[str] = []
    start = 0
    while start < len(text):
        window = text[start:start + maximum_tokens * 4]
        if start + len(window) == len(text) and _token_estimate(window) <= maximum_tokens:
            output.append(window)
            break
        low, high = 1, len(window)
        while low < high:
            middle = (low + high + 1) // 2
            if _token_estimate(window[:middle]) <= maximum_tokens:
                low = middle
            else:
                high = middle - 1
        boundary = window.rfind("\n", 0, low)
        end = boundary + 1 if boundary >= 0 else low
        output.append(window[:end])
        start += end
    return tuple(output)


def _source_text(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    value = re.sub(r"[\t ]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _path_key(path: Path) -> str:
    # Respect case-sensitive filesystems while normalizing Windows aliases.
    return unicodedata.normalize("NFKC", os.path.normcase(str(path)))


def _digest(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _source_text(value)
        key = normalize_surface(clean)
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result


__all__ = [
    "ExactTerm",
    "ExtractedDocument",
    "SUPPORTED_SUFFIXES",
    "collect_files",
    "extract_corpus",
    "extract_exact_surfaces",
    "extract_file",
    "normalize_surface",
    "source_sha256",
]
