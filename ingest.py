"""
Hafta 3 · Adim 1 — Belge yukleme + parcalama (chunking) + veritabanina yazma
config.DOCS_DIR icindeki tum .txt dosyalarini okur, uygun boyutta parcalara boler,
her parcayi embed'leyip SQLite'a yazar. RAG'in bilgi tabanini bu script kurar.

Calistir:
    python ingest.py            # docs/ klasorunu bastan isler (mevcut DB'yi siler)
"""

import os

import config
import db

# Desteklenen belge turleri. .txt/.md duz metin okunur; .pdf pypdf ile cozulur.
SUPPORTED = {".txt", ".md", ".pdf"}


def read_document(path) -> str:
    """Bir belgeyi uzantisina gore metne cevirir (.txt, .md, .pdf)."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader  # sadece PDF varsa yuklensin
        reader = PdfReader(str(path))
        sayfalar = [(sayfa.extract_text() or "") for sayfa in reader.pages]
        return "\n\n".join(sayfalar)
    raise ValueError(f"Desteklenmeyen belge turu: {path.name}")


def chunk_text(text: str, max_chars: int = None, min_chars: int = None,
               overlap: int = None) -> list[str]:
    """Bir belgeyi retrieval icin PARAGRAF-ONCELIKLI parcalara boler.

    Mantik: her paragraf (bos satirla ayrilmis blok) kural olarak KENDI parcasi
    olur -> tek bir konu baska konularla karismaz, embedding'i keskin kalir. Sadece
    cok kisa paragraflar (baslik gibi, < CHUNK_MIN_CHARS) tek basina birakilmaz,
    bir sonrakiyle birlestirilir. Cok uzun paragraf (> CHUNK_MAX_CHARS) ise
    CHUNK_OVERLAP kadar ortusmeyle pencerelere bolunur (cumleler ortadan kopmasin).

    Boyut parametreleri normalde config'ten gelir; bench_chunking.py bunlari
    disaridan vererek farkli parcalama ayarlarini AYNI kod uzerinde olcer
    (kiyas, uretimdekinden farkli bir parcalayiciyi olcmesin diye).
    """
    max_chars = config.CHUNK_MAX_CHARS if max_chars is None else max_chars
    min_chars = config.CHUNK_MIN_CHARS if min_chars is None else min_chars
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap

    paragraflar = [p.strip() for p in text.split("\n\n") if p.strip()]
    parcalar: list[str] = []
    tampon = ""  # doldurulmakta olan parca

    for p in paragraflar:
        # Tek basina cok uzun paragraf: once tamponu kapat, sonra pencerele.
        if len(p) > max_chars:
            if tampon:
                parcalar.append(tampon)
                tampon = ""
            start = 0
            while start < len(p):
                end = start + max_chars
                parcalar.append(p[start:end].strip())
                start = end - overlap
            continue

        if not tampon:
            tampon = p
        elif len(tampon) + 2 + len(p) <= max_chars:
            tampon += "\n\n" + p          # sigiyor -> ayni parcaya ekle
        else:
            parcalar.append(tampon)       # sigmiyor -> parcayi kapat, yenisini bas
            tampon = p

    if tampon:
        # Sona kalan parca cok kisaysa tek basina birakma, oncekine ekle.
        if parcalar and len(tampon) < min_chars:
            parcalar[-1] = (parcalar[-1] + "\n\n" + tampon).strip()
        else:
            parcalar.append(tampon)
    return parcalar


def ingest() -> None:
    # Bastan islemek icin varsa eski veritabanini sil.
    if config.DB_PATH.exists():
        os.remove(config.DB_PATH)

    if not config.DOCS_DIR.exists():
        raise SystemExit(f"Belge klasoru yok: {config.DOCS_DIR}")

    dosyalar = sorted(p for p in config.DOCS_DIR.iterdir()
                      if p.suffix.lower() in SUPPORTED)
    if not dosyalar:
        raise SystemExit(
            f"{config.DOCS_DIR} icinde desteklenen belge (.txt/.md/.pdf) bulunamadi.")

    conn = db.connect()
    db.init_db(conn)

    toplam_parca = 0
    for yol in dosyalar:
        metin = read_document(yol)
        parcalar = chunk_text(metin)
        if not parcalar:
            print(f"  {yol.name}: (bos/metin cikmadi, atlandi)")
            continue
        db.add_texts(conn, source=yol.name, texts=parcalar)
        toplam_parca += len(parcalar)
        print(f"  {yol.name}: {len(parcalar)} parca")

    belge_sayisi, parca_sayisi = db.count(conn)
    conn.close()
    print(f"\nBilgi tabani kuruldu: {belge_sayisi} belge, {parca_sayisi} parca -> {config.DB_PATH.name}")


if __name__ == "__main__":
    ingest()
