"""
Hafta 4 · Etkileşimli asistan (CLI)
Sürekli soru-cevap döngüsü: model bir kez yüklenir, sonra sen soru sordukça
RAG ile Türkçe cevaplar. Çıkmak için 'q' / 'çıkış' (Ctrl+C de olur).
Boş Enter oturumu KAPATMAZ; sadece nasıl soru sorulacağını hatırlatır.

Önce bilgi tabanı kurulmuş olmalı:
    python ingest.py
Sonra:
    python assistant.py

Her cevabın altında hangi belgeden geldiğini (kaynak) gösterir; böylece
asistanın uydurmadığını, gerçek bir belgeye dayandığını görebilirsin.
"""

import config
import db
import rag
from llm import load_chat

# Komutlar SADELESTIRILMIS halde tutulur (aksansiz, kucuk harf): "çıkış" da
# "CIKIS" da asagidaki normalize'dan sonra "cikis" olur, tek satir yeter.
CIKIS_KOMUTLARI = {"q", "quit", "exit", "cikis", "kapat"}

# Turkce'ye ozgu harflerin ASCII karsiligi (komut eslestirmesi icin).
_KATLAMA = str.maketrans("çğıöşü", "cgiosu")


def komut_normalize(s: str) -> str:
    """Kullanici girdisini komut karsilastirmasi icin sadelestirir.

    IKI ayri tuzak var, ikisi de tek basina cozum degil:

    1) Python'un str.lower()'i Turkce'nin noktali/noktasiz I kuralini bilmez:
       "ÇIKIŞ".lower() -> "çikiş" (noktali i). Onun icin once I->ı, İ->i.
    2) Ama yalnizca bunu yapinca ASCII yazim bozulur: "CIKIS" -> "cıkıs" olur ve
       "cikis" ile eslesmez. Turkce klavyesi olmayan kullanici Caps Lock'ta
       cikamaz hale gelir.

    Cozum ikisini birlestirmek: once Turkce'ye uygun kucult, sonra aksanlari
    ASCII'ye katla. Boylece "çıkış", "ÇIKIŞ", "cikis", "CIKIS", "Çıkış" -> hepsi
    "cikis" olur.
    """
    return s.replace("I", "ı").replace("İ", "i").lower().translate(_KATLAMA)


def main() -> None:
    if not config.DB_PATH.exists():
        raise SystemExit("Bilgi tabani yok. Once calistir:  python ingest.py")

    print("Deniz Yildizi Resort & Casino — Asistan")
    print("Model yukleniyor (ilk sefer ~10-15 sn)...")
    load_chat()  # modeli onceden yukle ki ilk soru hizli cevaplansin

    conn = db.connect()
    belge, parca = db.count(conn)
    print(f"Hazir. Bilgi tabani: {belge} belge, {parca} parca.")
    print("Soru sor (cikmak icin 'q').")

    while True:
        try:
            soru = input("\nSoru> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGorusuruz!")
            break

        if komut_normalize(soru) in CIKIS_KOMUTLARI:
            print("Gorusuruz!")
            break

        # Bos Enter artik CIKIS DEGIL: kazara basilan Enter oturumu (ve GPU'daki
        # yuklu modeli) kapatmasin. Bos girdiyi rag.answer zaten kapida yakalar;
        # burada onu cagirip ayni mesaji gostererek tek bir davranis kaynagi
        # tutuyoruz (mesaj iki yerde kopyalanmiyor).
        if not soru:
            print(rag.BOS_SORU_MESAJI)
            continue

        cevap, hits = rag.answer(soru, conn=conn)
        print(f"\n{cevap}")

        # Kaynak(lar): en benzer parcanin geldigi dosya(lar)
        kaynaklar = []
        for _, _, _, source in hits:
            if source not in kaynaklar:
                kaynaklar.append(source)
        print(f"  (kaynak: {', '.join(kaynaklar)})")

    conn.close()


if __name__ == "__main__":
    main()
