"""
Parcalama x embedding taramasi — "temsil e5'e gore mi ayarlanmis?" sorusuna cevap.

NEDEN BU TEST VAR: bench_embed.py, e5-small ile qwen3-embedding-0.6b'yi
karsilastirdi ve berabere bitti. Ama o kiyas, parcalari CHUNK_MAX_CHARS=350 ile
yapilmis bir bilgi tabani uzerinde yapti — ve bu ayar e5-small icin optimize
edilmisti. Qwen3-Embedding uzun baglam icin egitilmis bir model; 350 karakterlik
parcalar onun guclu oldugu yeri hic kullanmiyor olabilir. Yani kiyas, veriyi bir
modelin lehine hazirlamis olabilir.

Bu script ayni iki modeli DORT farkli parcalama ayarinda olcer.

OLCUM TUZAGI (ve cozumu): parca buyudukce hit@3 KENDILIGINDEN yukselir, cunku
3 parca bilgi tabaninin daha buyuk bir kismini kaplar. 350 karakterlik 3 parca
~1050 karakter baglam demektir; 1200 karakterlik 3 parca ~3600. Bunlar ayni sey
degildir. Bu yuzden iki metrik birden raporlanir:
  - hit@3          : ham (yaniltici, referans icin)
  - hit@k (butce)  : k, getirilen toplam karakter ~1050'de sabit kalacak sekilde
                     secilir -> LLM'e giden baglam esit, kiyas adil.

Calistirma:  python bench_chunking.py
"""

import numpy as np

import config
from bench_embed import SORULAR, embed_ile, sadelestir
from ingest import SUPPORTED, chunk_text, read_document

# (etiket, max_chars, min_chars, overlap) — None => belge hic bolunmez
AYARLAR = [
    ("350/120/80 (mevcut)", 350, 120, 80),
    ("700/200/120",         700, 200, 120),
    ("1200/300/150",       1200, 300, 150),
    ("belge butunu",       None, None, None),
]

BAGLAM_BUTCESI = 1050  # karakter — mevcut kurulumun getirdigi baglam (3 x 350)


def parcala(max_c, min_c, ovl) -> tuple[list[str], list[str]]:
    """docs/ klasorunu verilen ayarla parcalar. (metinler, kaynaklar) dondurur."""
    metinler, kaynaklar = [], []
    for yol in sorted(p for p in config.DOCS_DIR.iterdir() if p.suffix.lower() in SUPPORTED):
        ham = read_document(yol)
        parcalar = [ham.strip()] if max_c is None else chunk_text(ham, max_c, min_c, ovl)
        metinler.extend(parcalar)
        kaynaklar.extend([yol.name] * len(parcalar))
    return metinler, kaynaklar


def siralar(soru_vecs: np.ndarray, parca_vecs: np.ndarray, parca_metin: list[str]) -> list:
    """Her soru icin: kanit dizesini iceren ilk parcanin sirasi (1'den baslar)."""
    cikti = []
    for i, t in enumerate(SORULAR):
        sira = np.argsort(-(parca_vecs @ soru_vecs[i]))
        kanit = sadelestir(t["kanit"])
        cikti.append(next((r + 1 for r, idx in enumerate(sira)
                           if kanit in sadelestir(parca_metin[idx])), None))
    return cikti


def metrikler(sr: list, k_butce: int) -> dict:
    n = len(sr)
    return {
        "hit3": sum(1 for y in sr if y and y <= 3),
        "hitb": sum(1 for y in sr if y and y <= k_butce),
        "mrr": sum((1.0 / y) if y else 0.0 for y in sr) / n,
    }


def main() -> None:
    # Vektorler embed_texts icinde zaten birim uzunluga getiriliyor (normalize=True),
    # bu yuzden nokta carpimi dogrudan kosinus benzerligidir.
    sorular = [t["soru"] for t in SORULAR]
    print("e5-small ile sorular vektorlestiriliyor...")
    e5_soru = embed_ile("st", sorular, "query")
    print("qwen3-embedding yukleniyor ve sorular vektorlestiriliyor...")
    qw_soru = embed_ile("foundry", sorular, "query")

    n = len(SORULAR)
    print(f"\n{n} soru · her ayar icin iki model\n")
    baslik = (f"{'parcalama':<22}{'parca':>6}{'ort.uzn':>8}{'k(butce)':>9}"
              f"{'e5 hit@3':>10}{'e5 hit@k':>10}{'e5 MRR':>9}"
              f"{'qw hit@3':>10}{'qw hit@k':>10}{'qw MRR':>9}")
    print(baslik)
    print("-" * len(baslik))

    for etiket, mx, mn, ov in AYARLAR:
        metin, _ = parcala(mx, mn, ov)
        ort_uzn = sum(len(t) for t in metin) / len(metin)
        k_butce = max(1, round(BAGLAM_BUTCESI / ort_uzn))

        e5_parca = embed_ile("st", metin, "passage")
        qw_parca = embed_ile("foundry", metin, "passage")

        e = metrikler(siralar(e5_soru, e5_parca, metin), k_butce)
        q = metrikler(siralar(qw_soru, qw_parca, metin), k_butce)

        satir = f"{etiket:<22}{len(metin):>6}{ort_uzn:>8.0f}{k_butce:>9}"
        for m in (e, q):
            satir += f"{m['hit3']}/{n}".rjust(10)
            satir += f"{m['hitb']}/{n}".rjust(10)
            satir += f"{m['mrr']:.3f}".rjust(9)
        print(satir)

    print("\nhit@k(butce): getirilen toplam baglam ~1050 karakterde sabit tutulur —")
    print("parca buyudukce hit@3'un kendiliginden yukselmesi bu sutunda olmaz.")
    print("MRR: kanitin sirasinin tersinin ortalamasi (1.0 = hep 1. sirada).")


if __name__ == "__main__":
    main()
