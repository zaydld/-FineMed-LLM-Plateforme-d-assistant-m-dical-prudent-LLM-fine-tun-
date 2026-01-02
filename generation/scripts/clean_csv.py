import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IN_DIR = BASE_DIR / "reponses"

INPUT = IN_DIR / "reponses_llms.csv"
OUTPUT = IN_DIR / "reponses_llms_structured.csv"

def one_line(s: str) -> str:
    if s is None:
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n").strip()

# 🔹 lecture inchangée
with open(INPUT, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

rows.sort(key=lambda r: (r.get("id_cas",""), r.get("modele",""), int(r.get("sample_id","0"))))

# 🔴 SEULE MODIF ICI → utf-8-sig
with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for r in rows:
        r["prompt"] = one_line(r.get("prompt",""))
        r["reponse_modele"] = one_line(r.get("reponse_modele",""))
        writer.writerow(r)

print("✅ OK :", OUTPUT)
