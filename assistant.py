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

CIKIS_KOMUTLARI = {"q", "quit", "exit", "cikis", "çıkış", "kapat"}


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

        if soru.lower() in CIKIS_KOMUTLARI:
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
