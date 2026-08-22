"""
Site'deki "Deneyin" arama kutusu icin GERCEK calistirma kaydi uretir.

Neden boyle: yayinlanan sayfa statik; senin GPU'ndaki Foundry Local modeline
ulasamaz. Sayfada soru-cevap denenebilsin diye sorulari BURADA, gercek sistemde
calistirip cevabi, kaynak dosyalari, benzerlik skorlarini ve sureyi kaydediyoruz.
Sayfa bu kaydi okur; hicbir sey uydurulmaz, hicbir sey baska bir modele
sorulmaz.

Cikti: site/kayitlar.json   (site/index.html icine gomulur)
Calistirma: python site/kayit_uret.py
"""

import json
import sys
import time
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))

import config          # noqa: E402
import db              # noqa: E402
import rag             # noqa: E402
from llm import load_chat  # noqa: E402

# Bilgi tabaninda cevabi OLAN sorular + bilerek OLMAYANLAR.
# Dogrudan, dolayli ve gunluk dille yazilmis varyantlari birlikte tutuyoruz ki
# sayfadaki arama gercek kullanim cesitliligini yansitsin.
SORULAR = [
    # --- yeme icme ---
    "Kahvaltı kaçta başlar ve biter?",
    "Kahvaltıyı en geç kaçta yiyebilirim?",
    "Akşam à la carte İtalyan restoranı var mı?",
    "Akşam et yemek istiyorum, uygun bir restoran var mı?",
    "Denize karşı bir şeyler içebileceğim sahil barı var mı?",
    # --- odalar ---
    "Ocean Suite odasında jakuzi var mı?",
    "Otelde toplam kaç oda var?",
    "Odalarda kasa var mı?",
    # --- havuz & plaj ---
    "Otelde aquapark var mı?",
    "Otel toplam kaç havuzu var?",
    # --- spa & aktivite ---
    "Spa merkezinde Türk hamamı var mı?",
    "Otelde casino var mı?",
    "Çocuğumla yapabileceğim aktiviteler neler?",
    # --- ulasim ---
    "Ercan Havalimanı otele ne kadar uzakta?",
    "Uçaktan indim, otele nasıl ulaşabilirim?",
    # --- politika & genel ---
    "Odaya evcil hayvan getirebilir miyim?",
    "Otel hangi yıl hizmete açıldı?",
    "Balayı ayrıcalığından yararlanmak için ne gerekiyor?",
    # --- bilgi tabaninda YOK (guvenli basarisizlik ornekleri) ---
    "Otelde bowling salonu var mı?",
    "Bugün hava nasıl olacak?",
    "Otelde helikopter pisti var mı?",
]


def main() -> None:
    if not config.DB_PATH.exists():
        raise SystemExit("Once calistir:  python ingest.py")

    print("Model yukleniyor...")
    load_chat()
    conn = db.connect()
    belge, parca = db.count(conn)

    # ISINMA TURU (kaydedilmez): ilk cagri modelin ilk kez calismasinin
    # maliyetini tasiyor — olcumde 27 sn gorunmustu, oysa yerlesik hal ~2.5 sn.
    # O sayiyi sayfaya yazmak sistemi oldugundan yavas gosterirdi.
    print("Isinma turu (kaydedilmiyor)...")
    rag.answer("Merhaba, otel hakkinda bilgi verir misin?", conn=conn)

    kayitlar = []
    for i, soru in enumerate(SORULAR, 1):
        t0 = time.perf_counter()
        cevap, hits = rag.answer(soru, conn=conn)
        sure = time.perf_counter() - t0

        kaynaklar = []
        for skor, _, _, kaynak in hits:
            kaynaklar.append({"dosya": kaynak, "skor": round(float(skor), 3)})

        kayitlar.append({
            "soru": soru,
            "cevap": cevap.strip(),
            "kaynaklar": kaynaklar,
            "sure": round(sure, 2),
        })
        print(f"  [{i:2}/{len(SORULAR)}] {sure:5.2f} sn  {soru[:52]}")

    conn.close()

    cikti = {
        "uretim": {
            "belge": belge,
            "parca": parca,
            "chat_modeli": config.CHAT_MODEL,
            "embedding": config.EMBED_MODEL,
            "top_k": config.TOP_K,
        },
        "kayitlar": kayitlar,
    }

    yol = KOK / "site" / "kayitlar.json"
    yol.write_text(json.dumps(cikti, ensure_ascii=False, indent=2), encoding="utf-8")

    ort = sum(k["sure"] for k in kayitlar) / len(kayitlar)
    print(f"\n{len(kayitlar)} kayit yazildi -> {yol}")
    print(f"Ortalama cevap suresi: {ort:.2f} sn")


if __name__ == "__main__":
    main()
