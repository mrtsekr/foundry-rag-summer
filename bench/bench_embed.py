"""
Embedding arka uclarinin kiyasi — SADECE retrieval (LLM calismaz).

SORU: Bu proje icin hangi embedding modeli daha uygun?
  A) "st"      -> multilingual-e5-small   (sentence-transformers, 384 boyut)
  B) "foundry" -> qwen3-embedding-0.6b    (Foundry Local, 1024 boyut)

NEDEN BU OLCUM: eval.py'nin cevap dogrulugu temperature=0.7 yuzunden turdan
tura oynar; tek bir eval turuna bakip model secmek yaniltir. Retrieval ise
DETERMINISTIK: ayni soru + ayni korpus her zaman ayni siralamayi verir. Model
secimi bu yuzden burada, tekrarlanabilir sekilde yapilir.

Korpus URETIM ayarlariyla kurulur (config.CHUNK_*): karar dagitilan sistem icin
verilecegi icin olcum de dagitilan ayarda yapilmalidir.

Calistirma:  python bench_embed.py
"""

import sys
from pathlib import Path

# Bu dosya bench/ altinda; proje modulleri (config, db, ...) kokte duruyor.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import unicodedata

import numpy as np

import config
import rag_core
from ingest import chunk_text, read_document

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


def sadelestir(s: str) -> str:
    """Turkce'ye uygun kucuk harfe cevir (I/i sorunu dahil) ve normalize et."""
    s = s.replace("I", "ı").replace("İ", "i")
    return unicodedata.normalize("NFC", s).lower()


def korpus_kur() -> tuple[list[str], list[str]]:
    """docs/ klasorunu URETIM chunk ayarlariyla parcalara boler.

    rag_store.db'ye hic dokunulmaz: veritabani o an baska bir arka uca ait
    olabilir. Ayni parcalama kodu kullanildigi icin sonuc uretimdekiyle birebir.
    """
    metin, kaynak = [], []
    for yol in sorted(p for p in config.DOCS_DIR.iterdir()
                      if p.suffix.lower() in {".txt", ".md", ".pdf"}):
        for parca in chunk_text(read_document(yol), config.CHUNK_MAX_CHARS,
                                config.CHUNK_MIN_CHARS, config.CHUNK_OVERLAP):
            metin.append(parca)
            kaynak.append(yol.name)
    return metin, kaynak


def olc(ad, soru_vecs, parca_vecs, parca_metin, parca_kaynak, sure) -> dict:
    """Bir embedding arka ucunu olcer."""
    chunk_hit = {1: 0, 3: 0, 5: 0}   # kanit ilk k parcada mi
    dosya_hit3 = 0                   # dogru dosya ilk 3'te mi (eval.py ile ayni)
    ters_siralar, bantlar, siralar = [], [], []

    for i, t in enumerate(SORULAR):
        skorlar = parca_vecs @ soru_vecs[i]
        sira = np.argsort(-skorlar)
        kanit = sadelestir(t["kanit"])

        yer = next((r + 1 for r, idx in enumerate(sira)
                    if kanit in sadelestir(parca_metin[idx])), None)
        siralar.append(yer)
        for k in (1, 3, 5):
            if yer is not None and yer <= k:
                chunk_hit[k] += 1
        # MRR: dogru parca ne kadar ustteyse o kadar iyi (1/sira). Tek bir
        # "ilk-3'te mi" evet/hayirindan daha ince bir sinyal verir.
        ters_siralar.append(1.0 / yer if yer else 0.0)

        if any(parca_kaynak[idx] in t["kaynak"] for idx in sira[:3]):
            dosya_hit3 += 1

        ilk5 = skorlar[sira[:min(5, len(sira))]]
        bantlar.append(float(ilk5[0] - ilk5[-1]))

    return {
        "ad": ad, "chunk": chunk_hit, "dosya3": dosya_hit3,
        "mrr": float(np.mean(ters_siralar)), "bant": float(np.mean(bantlar)),
        "sure": sure, "siralar": siralar,
    }


def embed_ile(backend: str, metinler: list[str], kind: str) -> np.ndarray:
    """Metinleri VERILEN arka uc ile vektorlestirir; config'i gecici degistirir.

    Iki olcum script'inin (bench_embed, bench_chunking) ortak yardimcisi. Ikisi de
    ayni yoldan — rag_core.embed_texts — gectigi icin olculen sey uretimde
    calisan kodun ta kendisidir; on-ek kurallari (e5 query:/passage:, Qwen
    Instruct) orada bir kez tanimli, burada tekrarlanmaz.

    try/finally: olcum bittiginde ayar mutlaka eski haline doner, aksi halde
    sonraki cagrilar yanlis arka uca gider.
    """
    onceki = config.EMBED_BACKEND
    config.EMBED_BACKEND = backend          # rag_core.embed_texts bunu okur
    try:
        return np.asarray(rag_core.embed_texts(metinler, kind=kind), dtype=np.float32)
    finally:
        config.EMBED_BACKEND = onceki


def arka_ucu_olc(backend: str, ad: str, parca_metin, parca_kaynak) -> dict:
    """Verilen arka uctan hem parca hem soru vektorlerini alir ve olcer."""
    parca_vecs = embed_ile(backend, parca_metin, "passage")
    t0 = time.perf_counter()
    soru_vecs = embed_ile(backend, [t["soru"] for t in SORULAR], "query")
    sure = time.perf_counter() - t0
    return olc(ad, soru_vecs, parca_vecs, parca_metin, parca_kaynak, sure)


def main() -> None:
    parca_metin, parca_kaynak = korpus_kur()
    n = len(SORULAR)
    print(f"Korpus: {len(parca_metin)} parca (uretim ayarlari: max={config.CHUNK_MAX_CHARS} "
          f"min={config.CHUNK_MIN_CHARS} overlap={config.CHUNK_OVERLAP}) · {n} soru\n")

    print("[A] multilingual-e5-small (sentence-transformers)...")
    a = arka_ucu_olc("st", "e5-small (384)", parca_metin, parca_kaynak)

    print("[B] qwen3-embedding-0.6b (Foundry Local)...")
    b = arka_ucu_olc("foundry", "qwen3-emb (1024)", parca_metin, parca_kaynak)

    sonuclar = [a, b]
    print(f"\n{'':<26}" + "".join(f"{s['ad']:>20}" for s in sonuclar))
    print("-" * (26 + 20 * len(sonuclar)))
    for etiket, k in (("kanit ilk 1'de", 1), ("kanit ilk 3'te  ***", 3), ("kanit ilk 5'te", 5)):
        print(f"{etiket:<26}" + "".join(
            f"{s['chunk'][k]}/{n} (%{100*s['chunk'][k]/n:.0f})".rjust(20) for s in sonuclar))
    print(f"{'dogru DOSYA ilk 3te':<26}" + "".join(
        f"{s['dosya3']}/{n} (%{100*s['dosya3']/n:.0f})".rjust(20) for s in sonuclar))
    print(f"{'MRR (yuksek = iyi)':<26}" + "".join(f"{s['mrr']:.3f}".rjust(20) for s in sonuclar))
    print(f"{'skor bandi (ilk5 fark)':<26}" + "".join(f"{s['bant']:.3f}".rjust(20) for s in sonuclar))
    print(f"{'16 soruyu embedleme':<26}" + "".join(f"{s['sure']:.2f} sn".rjust(20) for s in sonuclar))

    print(f"\n{'Kanit kacinci sirada geldi':<48}" + "".join(f"{s['ad'][:12]:>14}" for s in sonuclar))
    print("-" * (48 + 14 * len(sonuclar)))
    for i, t in enumerate(SORULAR):
        hucreler = "".join((f"{s['siralar'][i]}." if s["siralar"][i] else "-").rjust(14)
                           for s in sonuclar)
        farkli = len({s["siralar"][i] for s in sonuclar}) > 1
        print(f"{'! ' if farkli else '  '}{t['soru'][:46]:<46}{hucreler}")

    print("\n'***' karar satiri: TOP_K=3 oldugu icin modelin gercekten gordugu bilgi budur.")
    print("'!' isaretli satirlar iki modelin AYRISTIGI sorular.")


if __name__ == "__main__":
    main()
