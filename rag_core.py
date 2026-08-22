"""
Hafta 2 · Adim 2 — Retrieval'in matematigi: embedding + kosinus benzerligi
Bu modul, projenin geri kalaninin kullanacagi yeniden-kullanilabilir
yardimci fonksiyonlari icerir:

  - get_embedder()      : embedding modelini bir kez yukleyip onbelleklar
  - embed_texts(...)    : metin listesini vektor dizisine cevirir
  - cosine_similarity() : iki vektorun ne kadar benzedigini [-1, 1] arasi
                          bir skorla verir (1 = ayni yon = cok benzer)

Dogrudan calistirirsan (python rag_core.py) kucuk bir gosterim yapar.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

import config


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Embedding modelini yukler. lru_cache sayesinde model program boyunca
    yalnizca BIR KEZ yuklenir; sonraki cagrilar ayni nesneyi geri verir."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(config.EMBED_MODEL, device=device)


def embed_texts(texts: list[str], kind: str = "passage", normalize: bool = True) -> np.ndarray:
    """Metin listesini (n adet) -> (n, EMBED_DIM) boyutlu float32 vektor dizisine
    cevirir.

    kind: "passage" (belge parcasi) ya da "query" (kullanici sorusu). e5 modeli
    bu ikisini farkli kodlar; dogru on-ek (config.EMBED_*_PREFIX) otomatik eklenir.
    Ayni metni yanlis on-ekle kodlamak retrieval kalitesini dusurur.

    normalize=True ise her vektor birim uzunluga (norm=1) getirilir; boylece
    kosinus benzerligi basit bir nokta carpimina indirgenir.
    """
    prefix = config.EMBED_QUERY_PREFIX if kind == "query" else config.EMBED_PASSAGE_PREFIX
    embedder = get_embedder()
    vecs = embedder.encode(
        [prefix + t for t in texts],
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Iki vektor arasindaki kosinus benzerligi.

    Formul:  cos = (a . b) / (|a| * |b|)
      - a . b   : nokta carpim (ayni yonu paylasan bilesenleri toplar)
      - |a|,|b| : vektorlerin uzunlugu (norm) — boylece uzunluk degil YON onemli olur

    Sonuc  1'e yakinsa: cok benzer,  0 civari: alakasiz,  -1'e yakin: zit.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _demo() -> None:
    """Adim 2 gosterimi: benzer cumleler yuksek, alakasiz cumle dusuk skor almali."""
    cumleler = [
        "Otelde kahvalti saat kacta baslar?",     # 0
        "Sabah kahvaltisi ne zaman aciliyor?",    # 1  (0 ile ayni anlam)
        "Havuz nerede bulunuyor?",                # 2  (alakasiz)
    ]
    # normalize=False veriyoruz ki cosine_similarity formulu uzunlugu kendisi
    # sadelestirsin — boylece formulun gercekten calistigini gormus oluruz.
    v = embed_texts(cumleler, normalize=False)

    print("Cumleler:")
    for i, c in enumerate(cumleler):
        print(f"  [{i}] {c}")

    print("\nKosinus benzerlik skorlari:")
    print(f"  [0] <-> [1] (ayni anlam)  : {cosine_similarity(v[0], v[1]):.3f}  <- yuksek olmali")
    print(f"  [0] <-> [2] (alakasiz)    : {cosine_similarity(v[0], v[2]):.3f}  <- dusuk olmali")
    print(f"  [1] <-> [2] (alakasiz)    : {cosine_similarity(v[1], v[2]):.3f}  <- dusuk olmali")
    print("\nAdim 2 tamam — anlamsal benzerligi sayiyla olcebiliyoruz. ✅")


if __name__ == "__main__":
    _demo()
