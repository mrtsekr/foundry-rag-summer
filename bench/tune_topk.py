"""
Hafta 5 · TOP_K ayari (retrieval taramasi)
Farkli TOP_K degerlerinde "dogru kaynak belge ilk-k parca arasinda mi?" oranini
olcer. SADECE retrieval calisir (LLM yuklenmez) -> cok hizli.

Amac: recall (dogru parcayi getirme) ile gurultu (alakasiz parca) arasindaki
dengeyi gorup config.TOP_K icin iyi bir deger secmek.

Once:  python ingest.py
Sonra: python tune_topk.py
"""

import sys
from pathlib import Path

# Bu dosya bench/ altinda; proje modulleri (config, db, ...) kokte duruyor.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import db
from eval import TESTLER  # ayni test setini kullan

# Sadece bilgi tabaninda OLAN sorular (kaynak != None) retrieval icin olculur.
OLCULEN = [t for t in TESTLER if t["kaynak"] is not None]


def beklenen_kaynaklar(t) -> set:
    """Test maddesinin kabul edilebilir kaynaklarini KUME olarak dondurur.

    Bazi bilgiler birden cok belgede geciyor, bu yuzden 'kaynak' alani bir
    dize ya da liste olabilir (bkz. eval.py). Ikisini de ayni sekilde ele
    almazsak liste gelen maddede karsilastirma coker.
    """
    k = t["kaynak"]
    return {k} if isinstance(k, str) else set(k)


def main() -> None:
    if not config.DB_PATH.exists():
        raise SystemExit("Once calistir:  python ingest.py")

    conn = db.connect()
    # ARTAN sirada olmali: asagidaki "ilk doyma noktasi" mantigi buna dayanir.
    k_degerleri = sorted([1, 2, 3, 4, 5, 6])

    print(f"Retrieval isabeti ({len(OLCULEN)} soru), farkli TOP_K icin:\n")
    print(f"{'TOP_K':>6}{'İsabet':>10}{'Oran':>8}")
    print("-" * 24)
    en_iyi = None
    for k in k_degerleri:
        isabet = 0
        for t in OLCULEN:
            hits = db.search(conn, t["soru"], top_k=k)
            kaynaklar = {src for _, _, _, src in hits}
            if beklenen_kaynaklar(t) & kaynaklar:
                isabet += 1
        oran = isabet / len(OLCULEN)
        print(f"{k:>6}{isabet:>7}/{len(OLCULEN)}{oran:>7.0%}")
        # Kesin ">" + artan tarama: isabetin doydugu EN KUCUK k tutulur.
        # (">=" yapilsaydi ya da liste azalan sirada olsaydi, en BUYUK k secilir
        # ve onerinin anlami tersine donerdi.)
        if en_iyi is None or isabet > en_iyi[1]:
            en_iyi = (k, isabet)

    conn.close()
    print(f"\nOnerilen TOP_K: {en_iyi[0]}  (isabet {en_iyi[1]}/{len(OLCULEN)}, "
          f"su an config.TOP_K = {config.TOP_K})")
    print("Not: isabetin doygunlastigi EN KUCUK k iyi bir secimdir")
    print("     (daha buyuk k, uretime alakasiz parca da tasiyip modeli sasirtabilir).")
    # Hangi sorular hala iskaliyor (en buyuk k'da) -> goster
    conn2 = db.connect()
    kmax = k_degerleri[-1]
    kalan = []
    for t in OLCULEN:
        hits = db.search(conn2, t["soru"], top_k=kmax)
        if not (beklenen_kaynaklar(t) & {src for _, _, _, src in hits}):
            kalan.append((t["soru"], " / ".join(sorted(beklenen_kaynaklar(t)))))
    conn2.close()
    if kalan:
        print(f"\nTOP_K={kmax}'da hala iskalanan sorular:")
        for soru, kaynak in kalan:
            print(f"  - [{kaynak}] {soru}")
    else:
        print(f"\nTOP_K={kmax}'da tum sorularin dogru kaynagi getiriliyor.")


if __name__ == "__main__":
    main()
