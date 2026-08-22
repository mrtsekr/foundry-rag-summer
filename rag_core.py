"""
Hafta 2 · Adim 2 — Retrieval'in matematigi: embedding + kosinus benzerligi
Bu modul, projenin geri kalaninin kullanacagi yeniden-kullanilabilir
yardimci fonksiyonlari icerir:

  - embed_texts(...)    : metin listesini vektor dizisine cevirir. HANGI modelle
                          yapilacagini config.EMBED_BACKEND belirler:
                            "foundry" -> qwen3-embedding-0.6b (Foundry Local)
                            "st"      -> multilingual-e5-small (sentence-transformers)
  - get_embedder()        : sentence-transformers modelini onbellekler
  - get_embedding_client(): Foundry embedding istemcisini onbellekler
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
    """sentence-transformers modelini yukler. lru_cache sayesinde model program
    boyunca yalnizca BIR KEZ yuklenir; sonraki cagrilar ayni nesneyi geri verir."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(config.ST_EMBED_MODEL, device=device)


@lru_cache(maxsize=1)
def get_embedding_client():
    """Foundry Local embedding istemcisini dondurur (model bir kez yuklenir).

    Sohbet modeli GPU'da oldugu icin embedding modelini CPU'da birakiyoruz
    (6 GB VRAM'in ~4.5 GB'ini qwen3-4b kullaniyor).
    """
    import foundry  # gec import: "st" arka ucunda Foundry'ye hic dokunulmaz
    model = foundry.load_model(config.FOUNDRY_EMBED_MODEL, gpu=False)
    return model.get_embedding_client()


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    """Satirlari birim uzunluga getirir; boylece kosinus = nokta carpimi olur."""
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def _embed_foundry(texts: list[str], kind: str, normalize: bool) -> np.ndarray:
    """Foundry Local (qwen3-embedding-0.6b) ile vektorlestirir.

    Qwen3-Embedding SORGUYA gorev tanimli bir on-ek ister, BELGEYI ham alir —
    e5'in query:/passage: ikilisinden farkli, asimetrik bir sema.
    """
    if kind == "query":
        texts = [config.FOUNDRY_QUERY_INSTRUCT + t for t in texts]

    istemci = get_embedding_client()
    cikti = []
    for i in range(0, len(texts), 16):  # parti parti: tek istekte cok metin sismesin
        yanit = istemci.generate_embeddings(texts[i:i + 16])
        cikti.extend(d.embedding for d in yanit.data)

    vecs = np.asarray(cikti, dtype=np.float32)
    return _normalize_rows(vecs) if normalize else vecs


def _embed_st(texts: list[str], kind: str, normalize: bool) -> np.ndarray:
    """sentence-transformers (multilingual-e5-small) ile vektorlestirir."""
    prefix = config.ST_QUERY_PREFIX if kind == "query" else config.ST_PASSAGE_PREFIX
    vecs = get_embedder().encode(
        [prefix + t for t in texts],
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_texts(texts: list[str], kind: str = "passage", normalize: bool = True) -> np.ndarray:
    """Metin listesini (n adet) -> (n, EMBED_DIM) boyutlu float32 vektor dizisine
    cevirir. Hangi modelin kullanilacagini config.EMBED_BACKEND belirler.

    kind: "passage" (belge parcasi) ya da "query" (kullanici sorusu). Iki arka uc
    de bu ayrimi kullanir ama FARKLI sekilde:
      - st      : "query: " / "passage: " on-ekleri (simetrik)
      - foundry : sorguya Instruct/Query on-eki, belgeye HIC on-ek yok (asimetrik)
    Ayni metni yanlis kind ile kodlamak retrieval kalitesini dusurur.

    normalize=True ise her vektor birim uzunluga (norm=1) getirilir; boylece
    kosinus benzerligi basit bir nokta carpimina indirgenir.
    """
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

    if config.EMBED_BACKEND == "foundry":
        return _embed_foundry(texts, kind, normalize)
    return _embed_st(texts, kind, normalize)


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
