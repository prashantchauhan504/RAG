from pathlib import Path

import fitz  # PyMuPDF


def main() -> None:
    out_path = Path("sample_for_chunking.pdf")

    doc = fitz.open()

    base = (
        "This is a test PDF for chunking. We want predictable content across pages. "
        "Chunking should create multiple pieces. "
    )

    # 3 pages, each long enough to create many chunks
    for page_num in [1, 2, 3]:
        unique = f"--- PAGE {page_num} UNIQUE MARKER ---\n"
        txt = unique + (base * 120)

        page = doc.new_page()
        rect = fitz.Rect(50, 50, 560, 750)  # left, top, right, bottom
        page.insert_textbox(rect, txt, fontsize=10)

    if out_path.exists():
        out_path.unlink()
    doc.save(str(out_path))
    doc.close()

    print("Wrote", out_path.resolve())


if __name__ == "__main__":
    main()

