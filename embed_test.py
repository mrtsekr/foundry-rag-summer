"""
Hafta 2 · Adim 1 — Embedding "smoke test"
Amac: sentence-transformers gercekten calisiyor mu, bir cumleyi vektore
ceviriyor mu, ve BENZER cumleler BENZER vektor mu veriyor? Bunu goruruz.

Henuz RAG yok; sadece retrieval'in temel tasi olan "metin -> vektor"
donusumunu ve vektorlerin anlamsal yakinligi yakaladigini dogruluyoruz.

Calistir (venv aktifken):
    python embed_test.py

Ilk calistirmada model (~90 MB) internetten iner ve onbellege alinir;
sonraki calistirmalar tamamen offline olur.
"""

from sentence_transformers import SentenceTransformer
import torch

import config

# 1) Embedding modelini yukle. Ilk seferde iner, sonra onbellekten gelir.
#    GPU (CUDA) varsa otomatik kullanilir -> embedding daha hizli olur.
# NOT: bu Hafta-2 gosterimi bilerek DOGRUDAN sentence-transformers modelini
# kullanir (config.ST_EMBED_MODEL), aktif arka uctan bagimsiz olarak. Amaci
# "metin -> vektor" fikrini en yalin haliyle gostermek; uretim yolu icin
# rag_core.embed_texts()'e bak (o config.EMBED_BACKEND'i dinler).
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cihaz: {device}")
print(f"Model yukleniyor: {config.ST_EMBED_MODEL} ...")
model = SentenceTransformer(config.ST_EMBED_MODEL, device=device)

# 2) Test cumleleri: ilk ikisi ayni anlama gelir, ucuncu alakasizdir.
#    Beklenti: cumle1 <-> cumle2 benzerligi YUKSEK, digerleri DUSUK olmali.
cumleler = [
    "Otelde kahvalti saat kacta baslar?",     # 0
    "Sabah kahvaltisi ne zaman aciliyor?",    # 1  (0 ile ayni anlam)
    "Havuz nerede bulunuyor?",                # 2  (alakasiz)
]

# 3) Cumleleri vektore cevir. Cikti: (cumle_sayisi, boyut) seklinde bir dizi.
vektorler = model.encode(cumleler)

print("\n--- Sonuc ---")
print(f"Vektor dizisinin sekli (shape): {vektorler.shape}")
print(f"  -> {vektorler.shape[0]} cumle, her biri {vektorler.shape[1]} boyutlu vektor")
print(f"config.ST_EMBED_DIM ile uyumlu mu? {vektorler.shape[1] == config.ST_EMBED_DIM}")

# 4) Ilk vektorden kucuk bir kesit gosterelim (sadece nasil gorundugunu gormek icin).
print(f"\n0. cumlenin vektorunun ilk 8 degeri:\n  {vektorler[0][:8]}")

print("\nAdim 1 tamam — metinler vektore donusuyor. ✅")
print("Not: 'ne kadar benziyorlar' skorunu Adim 2'de (rag_core.py) hesaplayacagiz.")
