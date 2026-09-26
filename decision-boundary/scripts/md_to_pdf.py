"""Convert a Markdown file (with embedded local image refs) into a PDF.

Pipeline:
    Markdown  →  HTML (Python `markdown` lib, with table+code+toc extensions)
              →  PDF  (playwright headless Chromium print)

Usage:
    uv run python scripts/md_to_pdf.py docs/PROBES_THEORY.md docs/PROBES_THEORY.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright


# CSS targeting a printable A4-ish layout with figure-friendly settings.
CSS = """
@page {
    size: A4;
    margin: 18mm 16mm;
}
* { box-sizing: border-box; }
html { font-size: 11pt; }
body {
    font-family: -apple-system, "Segoe UI", "Inter", Arial, sans-serif;
    color: #1f2328;
    line-height: 1.55;
    max-width: 100%;
}
h1 {
    font-size: 26pt;
    margin-top: 0;
    margin-bottom: 0.4em;
    border-bottom: 2px solid #1f2328;
    padding-bottom: 0.2em;
}
h2 {
    font-size: 18pt;
    margin-top: 1.6em;
    margin-bottom: 0.5em;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 0.15em;
    page-break-after: avoid;
}
h3 {
    font-size: 14pt;
    margin-top: 1.3em;
    margin-bottom: 0.35em;
    page-break-after: avoid;
}
h4 { font-size: 12pt; margin-top: 1em; }

p { margin: 0.5em 0; }

code {
    font-family: "Consolas", "Menlo", "DejaVu Sans Mono", monospace;
    font-size: 0.92em;
    background: #f5f5f5;
    padding: 1px 5px;
    border-radius: 3px;
}
pre {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 10px 12px;
    overflow-x: auto;
    font-size: 9.5pt;
    line-height: 1.4;
    page-break-inside: avoid;
}
pre code { background: transparent; padding: 0; font-size: inherit; }

blockquote {
    border-left: 3px solid #d0d7de;
    padding: 4px 12px;
    color: #555;
    background: #fafafa;
    margin: 0.6em 0;
}

table {
    border-collapse: collapse;
    margin: 0.8em 0;
    font-size: 0.93em;
    page-break-inside: avoid;
    width: auto;
    max-width: 100%;
}
th, td {
    border: 1px solid #d0d7de;
    padding: 5px 10px;
    text-align: left;
    vertical-align: top;
}
th {
    background: #f6f8fa;
    font-weight: 600;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.6em auto;
    page-break-inside: avoid;
}

ul, ol { margin: 0.4em 0 0.4em 1.4em; }
li { margin: 0.15em 0; }

hr {
    border: none;
    border-top: 1px solid #d0d7de;
    margin: 1.5em 0;
}

a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def md_to_html(md_path: Path) -> str:
    """Convert markdown to HTML, resolving image paths to absolute file:// URIs."""
    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "attr_list"],
        extension_configs={"codehilite": {"guess_lang": False, "noclasses": True}},
    )

    # Rewrite relative image paths into absolute file:// URIs so Chromium can fetch them.
    base = md_path.parent.resolve()

    def absolutize(html: str) -> str:
        import re
        def repl(match: re.Match) -> str:
            url = match.group(1)
            if url.startswith(("http://", "https://", "file://", "/", "data:")):
                return match.group(0)
            abs_path = (base / url).resolve()
            return f'src="file:///{abs_path.as_posix()}"'
        return re.sub(r'src="([^"]+)"', repl, html)

    html_body = absolutize(html_body)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{md_path.stem}</title>
<style>{CSS}</style>
</head><body>
{html_body}
</body></html>"""


def html_to_pdf(html: str, out_path: Path, source_dir: Path) -> None:
    """Render HTML with Chromium and print to PDF.

    Chromium refuses to load file:// images when content comes from set_content() —
    set_content has no origin, so cross-origin policy blocks file:// fetches.
    Workaround: write the HTML next to source_dir and load it via goto(file://...).
    Then both the page and its images live in the file:// origin and load freely.
    """
    tmp_html = source_dir / "_md_to_pdf_tmp.html"
    tmp_html.write_text(html, encoding="utf-8")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            file_url = "file:///" + tmp_html.resolve().as_posix()
            page.goto(file_url, wait_until="networkidle")
            page.pdf(
                path=str(out_path),
                format="A4",
                margin={"top": "18mm", "bottom": "18mm",
                        "left": "16mm", "right": "16mm"},
                print_background=True,
            )
            browser.close()
    finally:
        tmp_html.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("md_path", type=Path, help="Source markdown file")
    ap.add_argument("pdf_path", type=Path, help="Output PDF path")
    args = ap.parse_args()

    if not args.md_path.exists():
        print(f"error: {args.md_path} does not exist", file=sys.stderr)
        return 1

    print(f"Converting {args.md_path} -> {args.pdf_path}")
    html = md_to_html(args.md_path)
    html_to_pdf(html, args.pdf_path, args.md_path.parent)
    size_kb = args.pdf_path.stat().st_size / 1024
    print(f"Saved {args.pdf_path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
