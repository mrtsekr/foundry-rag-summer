"""
Hafta 2 · Adim 3 — Kalici depo: SQLite semasi + embedding saklama
Belgeleri parcalara (chunk) bolup HER parcanin metnini ve embedding'ini
veritabanina yaziyoruz. Boylece program her acildiginda embedding'leri
yeniden hesaplamak yerine hazir okuyoruz.

Sema (iki tablo):
  documents(id, source, added_at)        -> her kaynak belge icin bir satir
  chunks(id, doc_id, ord, text,          -> her belge parcasi icin bir satir
         embedding, dim)                     (embedding = ham float32 baytlar)

"embedding"i BLOB olarak, numpy float32 baytlariyla saklariz. SQLite'ta vektor
tipi yoktur; bu yuzden vektoru bayta cevirip (vector_to_blob) yaziyor,
okurken geri ceviriyoruz (blob_to_vector).

Dogrudan calistir (python db.py): semayi kurar, kucuk bir ornek veri ekler,
geri okur ve bir soruya en benzer parcayi bulur (Adim 1+2+3 birlikte).
"""

import sqlite3
from datetime import datetime, timezone

import numpy as np

import config
from rag_core import embed_texts, cosine_similarity


# --------------------------------------------------------------------------
# Baglanti ve sema
# --------------------------------------------------------------------------
def connect(db_path=config.DB_PATH) -> sqlite3.Connection:
    """Veritabani baglantisi acar. Foreign key kisitlarini da etkinlestirir."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Tablolar yoksa olusturur (varsa dokunmaz) ve eksik sutunu tamamlar."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            source   TEXT NOT NULL,          -- belgenin adi/yolu
            added_at TEXT NOT NULL,          -- eklenme zamani (ISO 8601)
            yuklenen INTEGER NOT NULL DEFAULT 0   -- 1 = kullanici ekledi
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id    INTEGER NOT NULL,       -- hangi belgeden geldigi
            ord       INTEGER NOT NULL,       -- belge icindeki sira (0,1,2,...)
            text      TEXT NOT NULL,          -- parcanin duz metni
            embedding BLOB NOT NULL,          -- float32 vektor (ham baytlar)
            dim       INTEGER NOT NULL,       -- vektor boyutu (dogrulama icin)
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        """
    )
    # Daha once kurulmus bir veritabaninda 'yuklenen' sutunu yoktur; CREATE
    # TABLE IF NOT EXISTS onu eklemez. Eksikse burada tamamliyoruz ki eski
    # rag_store.db'ler yeniden kurulmak zorunda kalmasin.
    sutunlar = {r[1] for r in conn.execute("PRAGMA table_info(documents);")}
    if "yuklenen" not in sutunlar:
        conn.execute("ALTER TABLE documents ADD COLUMN yuklenen INTEGER NOT NULL DEFAULT 0;")
    conn.commit()


# --------------------------------------------------------------------------
# Vektor <-> BLOB donusumu
# --------------------------------------------------------------------------
def vector_to_blob(vec: np.ndarray) -> bytes:
    """numpy vektorunu SQLite'a yazilabilir ham baytlara cevirir (float32)."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Veritabanindan okunan ham baytlari tekrar numpy float32 vektore cevirir."""
    return np.frombuffer(blob, dtype=np.float32)


# --------------------------------------------------------------------------
# Yazma
# --------------------------------------------------------------------------
def add_document(conn: sqlite3.Connection, source: str, yuklenen: bool = False) -> int:
    """Bir belge kaydi ekler ve yeni belgenin id'sini dondurur.

    yuklenen=True: belgeyi kullanici uygulamadan ekledi. Depoyla gelen
    korpustan ayirt edilebilsin diye isaretleniyor.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO documents (source, added_at, yuklenen) VALUES (?, ?, ?);",
        (source, now, 1 if yuklenen else 0),
    )
    conn.commit()
    return cur.lastrowid


def add_chunk(conn, doc_id: int, ord: int, text: str, embedding: np.ndarray) -> int:
    """Bir belge parcasini (metin + embedding) ekler, chunk id'sini dondurur."""
    cur = conn.execute(
        "INSERT INTO chunks (doc_id, ord, text, embedding, dim) VALUES (?, ?, ?, ?, ?);",
        (doc_id, ord, text, vector_to_blob(embedding), int(embedding.shape[0])),
    )
    conn.commit()
    return cur.lastrowid


def add_texts(conn, source: str, texts: list[str], yuklenen: bool = False) -> int:
    """Kolaylik: bir kaynak + metin parcalari listesini tek seferde embed'leyip
    veritabanina yazar. Belge id'sini dondurur."""
    doc_id = add_document(conn, source, yuklenen=yuklenen)
    vecs = embed_texts(texts)  # (n, EMBED_DIM), normalize edilmis
    for i, (text, vec) in enumerate(zip(texts, vecs)):
        add_chunk(conn, doc_id, i, text, vec)
    return doc_id


# --------------------------------------------------------------------------
# Okuma
# --------------------------------------------------------------------------
def all_chunks(conn) -> list[tuple[int, str, np.ndarray]]:
    """Tum parcalari (id, text, vektor) uclusu listesi olarak dondurur."""
    rows = conn.execute("SELECT id, text, embedding FROM chunks;").fetchall()
    return [(cid, text, blob_to_vector(blob)) for (cid, text, blob) in rows]


def yuklenen_belgeler(conn) -> list[tuple[str, int]]:
    """Kullanicinin ekledigi belgeleri (ad, parca_sayisi) olarak dondurur."""
    rows = conn.execute(
        "SELECT d.source, COUNT(c.id) "
        "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
        "WHERE d.yuklenen = 1 GROUP BY d.id ORDER BY d.id;"
    ).fetchall()
    return [(src, n) for (src, n) in rows]


def yuklenenleri_sil(conn) -> int:
    """Kullanicinin ekledigi TUM belgeleri siler, kac belge silindigini
    dondurur. Parcalar ON DELETE CASCADE ile birlikte gider; depoyla gelen
    korpus dokunulmadan kalir."""
    cur = conn.execute("DELETE FROM documents WHERE yuklenen = 1;")
    conn.commit()
    return cur.rowcount


def count(conn) -> tuple[int, int]:
    """(belge_sayisi, parca_sayisi) dondurur."""
    d = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0]
    c = conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
    return d, c


# --------------------------------------------------------------------------
# Retrieval (arama): bir soruya en benzer parcalari bul
# --------------------------------------------------------------------------
def search(conn, query: str, top_k: int = config.TOP_K) -> list[tuple[float, int, str, str]]:
    """Soruyu embed'ler, veritabanindaki TUM parcalarla kosinus benzerligine bakar
    ve en benzer ``top_k`` parcayi (skor, id, metin, kaynak_dosya) olarak, skora
    gore azalan sirada dondurur. RAG'in 'R' (retrieval) adimi budur."""
    (soru_vec,) = embed_texts([query], kind="query")
    rows = conn.execute(
        "SELECT c.id, c.text, c.embedding, d.source "
        "FROM chunks c JOIN documents d ON c.doc_id = d.id;"
    ).fetchall()

    # Arka uc degistiyse (or. e5-small 384 -> qwen3-embedding 1024) veritabanindaki
    # vektorler artik baska bir uzaya aittir. Bunu burada acikca yakalamazsak numpy
    # anlasilmaz bir boyut hatasi verir ya da (ayni boyutta farkli model olsaydi)
    # sessizce SACMA sonuclar dondururdu.
    if rows:
        kayitli_boyut = len(rows[0][2]) // 4  # float32 -> 4 bayt
        if kayitli_boyut != soru_vec.shape[0]:
            raise RuntimeError(
                f"Bilgi tabanindaki vektorler {kayitli_boyut} boyutlu, ama aktif embedding "
                f"modeli ({config.EMBED_BACKEND} / {config.EMBED_MODEL}) {soru_vec.shape[0]} "
                "boyut uretiyor.\nconfig.EMBED_BACKEND degistiyse bilgi tabani yeniden "
                "kurulmalidir:  python ingest.py"
            )

    puanli = [
        (cosine_similarity(soru_vec, blob_to_vector(blob)), cid, text, source)
        for (cid, text, blob, source) in rows
    ]
    puanli.sort(key=lambda x: x[0], reverse=True)
    return puanli[:top_k]


# --------------------------------------------------------------------------
# Gosterim: Adim 1 + 2 + 3 birlikte
# --------------------------------------------------------------------------
def _demo() -> None:
    """`python db.py` ile calisan gosterim: parca + embedding saklanip geri
    okunabiliyor mu.

    URETIM VERITABANINA DOKUNMAZ. Onceden config.DB_PATH'i silip ustune
    yaziyordu; dosyayi merakla calistiran biri bilgi tabanini kaybediyor ve
    `python ingest.py` calistirmak zorunda kaliyordu. Demo artik kendi ayri
    dosyasinda calisiyor ve bitince onu siliyor.
    """
    import os

    demo_yolu = config.PROJECT_DIR / "rag_store_demo.db"
    if demo_yolu.exists():
        os.remove(demo_yolu)

    conn = connect(demo_yolu)
    init_db(conn)

    # Kucuk bir "otel SSS" belgesi (3 parca).
    sss = [
        "Kahvalti her sabah saat 07:00 ile 10:30 arasinda restoranda servis edilir.",
        "Acik yuzme havuzu binanin arka bahcesinde olup 09:00-19:00 arasi aciktir.",
        "Check-out saati 12:00'dir; gec cikis icin resepsiyonla iletisime geciniz.",
    ]
    doc_id = add_texts(conn, source="otel_sss.txt", texts=sss)

    d, c = count(conn)
    print(f"Veritabani kuruldu: {d} belge, {c} parca (doc_id={doc_id}).")

    # Simdi bir soru soralim ve en benzer parcayi bulalim (basit retrieval).
    soru = "Sabah kahvaltiyi kacta yiyebilirim?"
    (soru_vec,) = embed_texts([soru], kind="query")

    print(f"\nSoru: {soru}")
    print("Parcalarin soruya benzerligi:")
    puanli = []
    for cid, text, vec in all_chunks(conn):
        skor = cosine_similarity(soru_vec, vec)
        puanli.append((skor, text))
    puanli.sort(reverse=True)  # en yuksek skor en ustte

    for skor, text in puanli:
        print(f"  {skor:.3f}  |  {text}")

    print(f"\nEN BENZER parca: {puanli[0][1]!r}")
    print("\nAdim 3 tamam — parcalar+embedding kalici saklaniyor ve geri okunup")
    print("bir soruya en yakin parca bulunabiliyor. Retrieval'in iskeleti hazir. ✅")

    conn.close()
    os.remove(demo_yolu)          # gosterim bitti, artigini birakma
    print(f"\n({demo_yolu.name} silindi; {config.DB_PATH.name} dokunulmadi.)")


if __name__ == "__main__":
    _demo()
