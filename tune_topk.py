"""
Hafta 5 · TOP_K ayari (retrieval taramasi)
Farkli TOP_K degerlerinde "dogru kaynak belge ilk-k parca arasinda mi?" oranini
olcer. SADECE retrieval calisir (LLM yuklenmez) -> cok hizli.

Amac: recall (dogru parcayi getirme) ile gurultu (alakasiz parca) arasindaki
dengeyi gorup config.TOP_K icin iyi bir deger secmek.

Once:  python ingest.py
Sonra: python tune_topk.py
"""

import config
import db
from eval import TESTLER  # ayni test setini kullan

# Sadece bilgi tabaninda OLAN sorular (kaynak != None) retrieval icin olculur.
OLCULEN = [t for t in TESTLER if t["kaynak"] is not None]


def main() -> None:
    if not config.DB_PATH.exists():
        raise SystemExit("Once calistir:  python ingest.py")

    conn = db.connect()
    k_degerleri = [1, 2, 3, 4, 5, 6]

    print(f"Retrieval isabeti ({len(OLCULEN)} soru), farkli TOP_K icin:\n")
    print(f"{'TOP_K':>6}{'İsabet':>10}{'Oran':>8}")
    print("-" * 24)
    en_iyi = None
    for k in k_degerleri:
        isabet = 0
        for t in OLCULEN:
            hits = db.search(conn, t["soru"], top_k=k)
            kaynaklar = {src for _, _, _, src in hits}
            if t["kaynak"] in kaynaklar:
                isabet += 1
        oran = isabet / len(OLCULEN)
        print(f"{k:>6}{isabet:>7}/{len(OLCULEN)}{oran:>7.0%}")
        if en_iyi is None or isabet > en_iyi[1]:
            en_iyi = (k, isabet)

    conn.close()
    # En dusuk k ile tam isabeti bul (fazla k gereksiz gurultu ekler)
    print("\nNot: isabetin doygunlastigi EN KUCUK k iyi bir secimdir")
    print("     (daha buyuk k, uretime alakasiz parca da tasiyip modeli sasirtabilir).")
    # Hangi sorular hala iskaliyor (en buyuk k'da) -> goster
    conn2 = db.connect()
    kmax = k_degerleri[-1]
    kalan = []
    for t in OLCULEN:
        hits = db.search(conn2, t["soru"], top_k=kmax)
        if t["kaynak"] not in {src for _, _, _, src in hits}:
            kalan.append((t["soru"], t["kaynak"]))
    conn2.close()
    if kalan:
        print(f"\nTOP_K={kmax}'da hala iskalanan sorular:")
        for soru, kaynak in kalan:
            print(f"  - [{kaynak}] {soru}")
    else:
        print(f"\nTOP_K={kmax}'da tum sorularin dogru kaynagi getiriliyor.")


if __name__ == "__main__":
    main()
