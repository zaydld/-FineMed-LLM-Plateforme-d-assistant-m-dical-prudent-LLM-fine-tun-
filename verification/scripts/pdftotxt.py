from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_SOURCES = BASE_DIR / "rag_sources"

def pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def process_folder(folder: Path):
    for pdf in folder.rglob("*.pdf"):
        txt_path = pdf.with_suffix(".txt")
        if txt_path.exists():
            continue  # déjà converti

        print(f"📄 Conversion: {pdf}")
        try:
            text = pdf_to_text(pdf)
            if len(text.strip()) < 300:
                print(f"⚠️ Texte trop court: {pdf.name}")
                continue

            txt_path.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"❌ Erreur {pdf.name}: {e}")

def main():
    for source in ["has", "oms"]:
        src_path = RAG_SOURCES / source
        if src_path.exists():
            process_folder(src_path)

    print("✅ Conversion PDF → TXT terminée")

if __name__ == "__main__":
    main()
