"""
Hafta 4 · Etkileşimli asistan (CLI)
Sürekli soru-cevap döngüsü: model bir kez yüklenir, sonra sen soru sordukça
RAG ile Türkçe cevaplar. Çıkmak için 'q', 'çıkış' veya boş Enter (Ctrl+C de olur).

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
    print("Soru sor (cikmak icin 'q' ya da bos Enter).")

    while True:
        try:
            soru = input("\nSoru> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGorusuruz!")
            break

        if not soru or soru.lower() in CIKIS_KOMUTLARI:
            print("Gorusuruz!")
            break

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
