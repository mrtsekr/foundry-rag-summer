"""
Embedding karsilastirmasi — SADECE retrieval (LLM calismaz, bu yuzden hizli).

Soru: Retrieval'i Foundry Local'in embedding modeline tasimali miyiz?
Karsilastirilan uc sistem:

  A) e5-small        : su anki kurulum (sentence-transformers, 384 boyut)
  B) qwen3-emb (ham) : Foundry Local qwen3-embedding-0.6b, on-eksiz (1024 boyut)
  C) qwen3-emb (instruct) : ayni model, Qwen3'un onerdigi sorgu on-eki ile

NEDEN CHUNK SEVIYESI OLCUYORUZ: eval.py'deki retrieval metrigi DOSYA bazlidir
("dogru belge geldi mi") ve bu IYIMSER bir olcumdur. Ornek: "kahvalti kacta
biter" sorusunda dogru dosya (yeme_icme.txt) ilk 3'e giriyor ama saatleri iceren
PARCA 5. sirada kaliyor -> metrik "OK" derken model cevabi hic gormuyor.
Burada her soru icin bir "kanit" dizesi tanimlanir (cevabin gectigi metin) ve
kanit getirilen parcalarin ICINDE mi diye bakilir. Gercegi bu olcer.

Calistirma:  python bench_embed.py     (once: python ingest.py)
"""

import time
import unicodedata

import numpy as np

import config
from ingest import chunk_text, read_document
from rag_core import embed_texts

# Her soru icin: cevabin dayandigi KANIT dizesi (dogru parcada gecmeli) ve
# kabul edilebilir kaynak dosya(lar). Kanitlar bilerek kisa/kok tutuldu ki
# Turkce cekim farklari eslesmeyi bozmasin.
SORULAR = [
    {"soru": "Kahvalti kacta baslar ve biter?",                          "kanit": "07:00",       "kaynak": {"yeme_icme.txt"}},
    {"soru": "Aksam a la carte Italyan restorani var mi?",               "kanit": "italyan",     "kaynak": {"yeme_icme.txt"}},
    {"soru": "Ocean Suite odasinda jakuzi var mi?",                      "kanit": "jakuzi",      "kaynak": {"odalar.txt"}},
    {"soru": "Otelde toplam kac oda var?",                               "kanit": "569",         "kaynak": {"otel_genel.txt"}},
    {"soru": "Ercan Havalimani otele ne kadar uzakta?",                  "kanit": "56,8",        "kaynak": {"ulasim.txt"}},
    {"soru": "Spa merkezinde Turk hamami var mi?",                       "kanit": "hamam",       "kaynak": {"spa.md"}},
    {"soru": "Otelde aquapark var mi?",                                  "kanit": "aquapark",    "kaynak": {"havuz_plaj.txt"}},
    {"soru": "Odaya evcil hayvan getirebilir miyim?",                    "kanit": "evcil hayvan","kaynak": {"politikalar.txt"}},
    {"soru": "Otelde casino var mi?",                                    "kanit": "casino",      "kaynak": {"aktiviteler.txt"}},
    {"soru": "Otel toplam kac havuzu var?",                              "kanit": "10 adet",     "kaynak": {"havuz_plaj.txt"}},
    {"soru": "Otel hangi yil hizmete acildi?",                           "kanit": "2018",        "kaynak": {"otel_genel.txt"}},
    {"soru": "Denize karsi bir seyler icebilecegim sahil bari var mi?",  "kanit": "sahil bar",   "kaynak": {"yeme_icme.txt"}},
    {"soru": "Cocugumla yapabilecegim aktiviteler neler?",               "kanit": "mini kul",    "kaynak": {"aktiviteler.txt"}},
    {"soru": "Balayi ayricaligindan yararlanmak icin ne gerekiyor?",     "kanit": "evlilik",     "kaynak": {"balayi.pdf", "politikalar.txt"}},
    {"soru": "Ucaktan indim, otele nasil ulasabilirim?",                 "kanit": "transfer",    "kaynak": {"ulasim.txt"}},
    {"soru": "Aksam et yemek istiyorum, uygun bir restoran var mi?",     "kanit": "steakhouse",  "kaynak": {"yeme_icme.txt"}},
]

# Qwen3-Embedding, sorgular icin gorev tanimli bir on-ek onerir; belgeler ham verilir.
QWEN_INSTRUCT = (
    "Instruct: Given a Turkish question about a hotel, retrieve the passage that answers it\n"
    "Query: "
)


def sadelestir(s: str) -> str:
    """Turkce'ye uygun kucuk harfe cevir (I/i sorunu dahil) ve normalize et."""
    s = s.replace("I", "ı").replace("İ", "i")
    return unicodedata.normalize("NFC", s).lower()


def birimle(M: np.ndarray) -> np.ndarray:
    """Satirlari birim uzunluga getirir; boylece kosinus = nokta carpimi olur."""
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def foundry_istemci():
    """qwen3-embedding-0.6b'yi Foundry Local'da yukler ve embedding istemcisini dondurur."""
    from foundry_local_sdk import Configuration, FoundryLocalManager

    FoundryLocalManager.initialize(Configuration(app_name="bench_embed"))
    model = FoundryLocalManager.instance.catalog.get_model("qwen3-embedding-0.6b")
    if not model.is_cached:
        print("  qwen3-embedding-0.6b indiriliyor (~495 MB, tek sefer)...")
        model.download(lambda p: print(f"\r    {p:.0f}%", end="", flush=True))
        print()
    model.load()
    return model.get_embedding_client()


def foundry_embed(istemci, metinler: list[str], toplu: int = 16) -> np.ndarray:
    """Metin listesini Foundry embedding modeliyle vektorlestirir (parti parti)."""
    cikti = []
    for i in range(0, len(metinler), toplu):
        yanit = istemci.generate_embeddings(metinler[i:i + toplu])
        cikti.extend(d.embedding for d in yanit.data)
    return birimle(np.asarray(cikti, dtype=np.float32))


def olc(ad: str, soru_vecs: np.ndarray, parca_vecs: np.ndarray,
        parca_metin: list[str], parca_kaynak: list[str], sure: float) -> dict:
    """Bir embedding sistemini olcer ve metrikleri dondurur."""
    chunk_hit = {1: 0, 3: 0, 5: 0}   # kanit ilk k parcada mi (GERCEK metrik)
    dosya_hit3 = 0                    # dogru dosya ilk 3'te mi (eval.py ile ayni)
    bantlar, siralar = [], []

    for i, t in enumerate(SORULAR):
        skorlar = parca_vecs @ soru_vecs[i]
        sira = np.argsort(-skorlar)
        kanit = sadelestir(t["kanit"])

        # Kanitin gectigi ilk parcanin kacinci sirada oldugu
        yer = next((r + 1 for r, idx in enumerate(sira)
                    if kanit in sadelestir(parca_metin[idx])), None)
        siralar.append(yer)
        for k in (1, 3, 5):
            if yer is not None and yer <= k:
                chunk_hit[k] += 1

        if any(parca_kaynak[idx] in t["kaynak"] for idx in sira[:3]):
            dosya_hit3 += 1

        ilk5 = skorlar[sira[:5]]
        bantlar.append(float(ilk5[0] - ilk5[-1]))

    n = len(SORULAR)
    return {
        "ad": ad, "n": n, "chunk": chunk_hit, "dosya3": dosya_hit3,
        "bant": float(np.mean(bantlar)), "sure": sure, "siralar": siralar,
    }


def main() -> None:
    # Korpusu docs/'tan DOGRUDAN kuruyoruz; rag_store.db'ye hic dokunmuyoruz.
    # Sebep: veritabani baska bir olcumun (chunking taramasi) araci olabilir ve
    # ingest.py onu silip yeniden yazar. Ayni parcalama kodu (ingest.chunk_text,
    # config ayarlariyla) kullanildigi icin sonuc uretimdekiyle birebir aynidir.
    # Chunk ayarlari BURADA sabit: bu kiyasin tek degiskeni EMBEDDING MODELI olmali.
    # config.CHUNK_* okunsaydi, es zamanli bir chunking taramasi bu olcumu sessizce
    # kaydirirdi (nitekim tarama sirasinda CHUNK_MAX_CHARS 1000'e cekilmisti ve
    # korpus 31 yerine 9 parcaya dusmustu).
    MAX_CHARS, MIN_CHARS, OVERLAP = 350, 120, 80

    parca_metin, parca_kaynak = [], []
    for yol in sorted(p for p in config.DOCS_DIR.iterdir()
                      if p.suffix.lower() in {".txt", ".md", ".pdf"}):
        for parca in chunk_text(read_document(yol), MAX_CHARS, MIN_CHARS, OVERLAP):
            parca_metin.append(parca)
            parca_kaynak.append(yol.name)

    sorular = [t["soru"] for t in SORULAR]
    print(f"Korpus: {len(parca_metin)} parca (docs/'tan, sabitlenmis uretim ayarlari: "
          f"max={MAX_CHARS} min={MIN_CHARS} overlap={OVERLAP}) · {len(SORULAR)} soru\n")

    sonuclar = []

    # --- A) Mevcut sistem: e5-small (uretimdeki gibi passage/query on-ekleriyle)
    print("[A] e5-small (mevcut)...")
    e5_parca = birimle(np.asarray(embed_texts(parca_metin, kind="passage"), dtype=np.float32))
    t0 = time.perf_counter()
    e5_soru = birimle(np.asarray(embed_texts(sorular, kind="query"), dtype=np.float32))
    sonuclar.append(olc("e5-small (384)", e5_soru, e5_parca, parca_metin,
                        parca_kaynak, time.perf_counter() - t0))

    # --- B/C) Foundry Local: qwen3-embedding-0.6b
    print("[B/C] qwen3-embedding-0.6b (Foundry Local) yukleniyor...")
    istemci = foundry_istemci()

    t0 = time.perf_counter()
    q_parca = foundry_embed(istemci, parca_metin)
    parca_sure = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_soru_ham = foundry_embed(istemci, sorular)
    sonuclar.append(olc("qwen3-emb ham (1024)", q_soru_ham, q_parca, parca_metin,
                        parca_kaynak, time.perf_counter() - t0))

    t0 = time.perf_counter()
    q_soru_ins = foundry_embed(istemci, [QWEN_INSTRUCT + s for s in sorular])
    sonuclar.append(olc("qwen3-emb instruct", q_soru_ins, q_parca, parca_metin,
                        parca_kaynak, time.perf_counter() - t0))

    # ---------------- Sonuc tablosu ----------------
    n = len(SORULAR)
    print(f"\n{'':<24}" + "".join(f"{s['ad']:>22}" for s in sonuclar))
    print("-" * (24 + 22 * len(sonuclar)))
    for etiket, k in (("kanit ilk 1'de", 1), ("kanit ilk 3'te  ***", 3), ("kanit ilk 5'te", 5)):
        satir = "".join(f"{s['chunk'][k]}/{n} (%{100*s['chunk'][k]/n:.0f})".rjust(22) for s in sonuclar)
        print(f"{etiket:<24}{satir}")
    print(f"{'dogru DOSYA ilk 3te':<24}" +
          "".join(f"{s['dosya3']}/{n} (%{100*s['dosya3']/n:.0f})".rjust(22) for s in sonuclar))
    print(f"{'skor bandi (ilk5 fark)':<24}" +
          "".join(f"{s['bant']:.3f}".rjust(22) for s in sonuclar))
    print(f"{'16 soruyu embedleme':<24}" +
          "".join(f"{s['sure']:.2f} sn".rjust(22) for s in sonuclar))
    print(f"\n(31 parcayi embedleme, Foundry: {parca_sure:.1f} sn — tek seferlik, ingest'te olur)")

    # Soru bazinda kanitin kacinci sirada geldigi (None = hic bulunamadi)
    print(f"\n{'Kanit kacinci sirada geldi':<46}" + "".join(f"{s['ad'][:10]:>13}" for s in sonuclar))
    print("-" * (46 + 13 * len(sonuclar)))
    for i, t in enumerate(SORULAR):
        hucreler = ""
        for s in sonuclar:
            y = s["siralar"][i]
            hucreler += (f"{y}." if y else "-").rjust(13)
        isaret = "  " if all(s["siralar"][i] and s["siralar"][i] <= 3 for s in sonuclar) else "! "
        print(f"{isaret}{t['soru'][:44]:<44}{hucreler}")
    print("\n'***' satiri karar satiri: TOP_K=3 oldugu icin modelin gercekten")
    print("gordugu bilgi budur. Digerleri baglam icin.")


if __name__ == "__main__":
    main()
