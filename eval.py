"""
Hafta 5 · Basit değerlendirme (evaluation)
Elle hazirlanmis bir soru-cevap seti uzerinde RAG'i olcer. Iki metrik:

  1) Cevap dogrulugu: cevap, beklenen anahtar kelime(ler)den en az birini iceriyor mu?
  2) Retrieval isabeti: dogru kaynak belge getirilen parcalar arasinda mi?

Bu ikisini ayirmak onemli: cevap yanlissa sorun retrieval'da mi (yanlis parca
geldi) yoksa uretimde mi (dogru parca geldi ama model yanlis cevapladi) anlariz.

Ayrica ucuncu bir bolum var: UC DURUMLAR (bos girdi, cok genel soru, konu disi
soru). Onlar dogrulugu degil DAYANIKLILIGI olcer ve yukaridaki yuzdelere
karismaz — ayri raporlanir.

Once:  python ingest.py
Sonra: python eval.py
"""

import unicodedata

import config
import db
import rag
from llm import load_chat

# Test seti: her madde bir soru, beklenen anahtar kelimeler ve dogru kaynak dosya.
#   - "anahtar": bu kelimelerden EN AZ BIRI cevapta (alt-dizi olarak) gecerse cevap
#     dogru sayilir. Bir sorunun birden fazla gecerli anahtari olabilir.
#   - "kaynak": dogru bilginin bulundugu belge. Bir dize ya da LISTE olabilir;
#     liste verilince getirilen parcalardan HERHANGI BIRI listede varsa isabet
#     sayilir. Sebep: bazi bilgiler birden fazla belgede gecer ve hangisinin
#     geldigi onemli degildir. Ornek: "Otelde casino var mi?" sorusunun cevabi
#     hem otel_genel.txt'te ("Otel bunyesinde genis bir casino ... bulunur")
#     hem aktiviteler.txt'te ("Tesiste ayrica buyuk bir casino (kumarhane)
#     bulunur") yazili. Tek kaynak beklerken asistan DOGRU cevap verdigi halde
#     metrik ISKA sayiyordu — olculen sey retrieval degil, sansti.
#     None ise bilgi tabaninda YOKTUR
#     (asistan uydurmayip "bilgi yok"/"bulunmamaktadir" demeli).
#
# DIKKAT — bazi anahtarlar bilerek KÖK'tur, tam kelime degil (or. "bulunm",
# "kabul edilm"). Turkce sondan eklemeli oldugu icin tek bir ifade cok farkli
# cekimlenir: "bulunmuyor / bulunmamaktadir / bulunmaz". Tam kelime yazsak
# bunlarin bir kismini KACIRIRDIK. Bu yuzden (tam bir Turkce stemmer yerine)
# hafif bir yontemle kök eslesmesi yapiyoruz. Bu bir yazim hatasi DEGILDIR.
#
# DIKKAT 2 — anahtarlarda tek basina "var" KULLANMIYORUZ. Once kullaniyorduk ve
# metrigi SISIRIYORDU: "en az bir anahtar gecsin" kurali yuzunden, model yanlis
# bir sey soylese bile cumlesinde "var" gectigi icin cevap DOGRU sayilabiliyordu.
# Artik her soru, o soruya OZGU terimi (jakuzi, hamam, aquapark, casino, sahil
# bar...) istiyor. Bu, olculen basariyi dusurur ama gercege yaklastirir.
# AYNI KURALIN GENEL HALI: SORUNUN ICINDE gecen bir kelime anahtar olamaz.
# Soru "...var mi?" ile bitiyorsa "var", "aktiviteler neler?" ise "aktivite"
# anahtari, model soruyu yankilamasi halinde bile gecer -> metrik siser.
# Anahtar, cevabin DOGRU oldugunu ayirt eden terim olmali.
#
# Not: bilerek DOLAYLI/parafraz sorular da var (retrieval'i zorlamak icin).
TESTLER = [
    # --- dogrudan sorular ---
    {"soru": "Kahvalti kacta baslar ve biter?", "anahtar": ["07:00", "11:00"], "kaynak": "yeme_icme.txt"},
    {"soru": "Aksam a la carte Italyan restorani var mi?", "anahtar": ["İtalyan"], "kaynak": "yeme_icme.txt"},
    {"soru": "Ocean Suite odasinda jakuzi var mi?", "anahtar": ["jakuzi"], "kaynak": "odalar.txt"},
    {"soru": "Otelde toplam kac oda var?", "anahtar": ["569"], "kaynak": "otel_genel.txt"},
    {"soru": "Ercan Havalimani otele ne kadar uzakta?", "anahtar": ["56"], "kaynak": "ulasim.txt"},
    {"soru": "Spa merkezinde Turk hamami var mi?", "anahtar": ["hamam"], "kaynak": "spa.md"},
    {"soru": "Otelde aquapark var mi?", "anahtar": ["aquapark", "su parkı"], "kaynak": "havuz_plaj.txt"},
    {"soru": "Odaya evcil hayvan getirebilir miyim?", "anahtar": ["kabul edilm", "hayır", "yok"], "kaynak": "politikalar.txt"},
    {"soru": "Otelde casino var mi?", "anahtar": ["casino", "kumarhane"],
     "kaynak": ["aktiviteler.txt", "otel_genel.txt"]},
    {"soru": "Otel toplam kac havuzu var?", "anahtar": ["10"], "kaynak": "havuz_plaj.txt"},
    {"soru": "Otel hangi yil hizmete acildi?", "anahtar": ["2018"], "kaynak": "otel_genel.txt"},
    # --- dolayli / parafraz sorular (retrieval'i zorlar) ---
    {"soru": "Denize karsi bir seyler icebilecegim sahil bari var mi?", "anahtar": ["sahil bar"], "kaynak": "yeme_icme.txt"},
    {"soru": "Cocugumla yapabilecegim aktiviteler neler?", "anahtar": ["çocuk", "oyun", "mini kulüp"], "kaynak": "aktiviteler.txt"},
    {"soru": "Balayi ayricaligindan yararlanmak icin ne gerekiyor?", "anahtar": ["evlilik", "cüzdan", "6 ay"],
     "kaynak": ["balayi.pdf", "politikalar.txt"]},   # kural iki belgede de yazili
    {"soru": "Ucaktan indim, otele nasil ulasabilirim?", "anahtar": ["transfer", "Ercan", "56"], "kaynak": "ulasim.txt"},
    {"soru": "Aksam et yemek istiyorum, uygun bir restoran var mi?", "anahtar": ["steak", "et restoran"], "kaynak": "yeme_icme.txt"},
    # --- bilgi tabaninda OLMAYAN (uydurmamali) ---
    {"soru": "Otelde bowling salonu var mi?", "anahtar": ["bilgi yok", "bulunm", "yok", "danış"], "kaynak": None},
]


# --- Uc durumlar (edge cases) ---
# Plan s.11 acikca istiyor: "empty query input, or very general questions".
# Bunlar bilerek AYRI olculur ve ana yuzdelere KARISMAZ; iki farkli soru
# soruyorlar: ana set "dogru cevabi biliyor mu?", burasi "beklenmedik girdide
# guvenli mi davraniyor?". Karistirsaydik headline sayisi ne artisa ne dususe
# dogru yorumlanabilirdi (5 kolay uc durum yuzdeyi sisirirdi).
#   - "bekle"   : cevapta gecmesi beklenen anahtar(lar)dan en az biri
#   - "hits_bos": True ise retrieval HIC calismamis olmali (kaynak listesi bos)
UC_DURUMLAR = [
    {"ad": "bos sorgu", "soru": "",
     "bekle": ["soru yazmadiniz"], "hits_bos": True,
     "neden": "Bos girdide retrieval ve uretim hic calismamali; sabit yonlendirme donmeli."},
    {"ad": "sadece bosluk", "soru": "   \t  ",
     "bekle": ["soru yazmadiniz"], "hits_bos": True,
     "neden": "Bosluk-only girdi de bos sayilmali (strip sonrasi ayni yol)."},
    {"ad": "cok genel soru", "soru": "Bana otelden bahset.",
     "bekle": ["otel", "resort", "deniz yıldızı", "bafra", "oda", "havuz"],
     "neden": "Genel soruda cevap uretmeli ama baglamdaki gercege dayanmali."},
    {"ad": "tek kelime", "soru": "havuz",
     "bekle": ["havuz"],
     "neden": "Tam cumle olmayan girdide bile dogru konuya gitmeli."},
    {"ad": "konu disi soru", "soru": "Bugun hava nasil olacak?",
     "bekle": ["bilgi yok", "bulunm", "resepsiyon", "yok"],
     "neden": "Bilgi tabani disi soruda uydurmamali; guvenli sekilde basarisiz olmali."},
]


# NOT: Ayni islev bench/bench_embed.py icinde de var. Bilerek kopyalandi:
# oradan almak, LLM yigini yuklemeyen bir olcum betigini eval.py
# uzerinden llm/foundry'ye baglardi. Ikisi AYNI kalmali, yoksa iki
# olcumun sayilari kiyaslanamaz hale gelir.
def sadelestir(s: str) -> str:
    """Turkce'ye uygun kucuk harfe cevir (I/ı, İ/i dahil) ve normalize et."""
    s = s.replace("I", "ı").replace("İ", "i")
    return unicodedata.normalize("NFC", s).lower()


def uc_durumlari_calistir(conn) -> tuple[int, int]:
    """UC_DURUMLAR listesini calistirir ve (gecen, toplam) dondurur.

    Ana tablodan ayri basilir: burada olculen sey dogruluk degil DAYANIKLILIK.
    """
    print("\n" + "=" * 62)
    print("UC DURUMLAR — bos girdi / cok genel / konu disi (plan s.11)")
    print("=" * 62)

    gecen = 0
    for u in UC_DURUMLAR:
        cevap, hits = rag.answer(u["soru"], conn=conn)
        cl = sadelestir(cevap)
        ok = any(sadelestir(b) in cl for b in u["bekle"])
        # Bos girdide kaynak listesi de bos olmali: retrieval'a hic gidilmedigin
        # kaniti budur (aksi halde model rastgele baglamla cevap uydurabilirdi).
        if u.get("hits_bos") and hits:
            ok = False
        gecen += ok

        print(f"\n[{'OK' if ok else 'YANLIS'}] {u['ad']}   girdi: {u['soru']!r}")
        print(f"      beklenen : {u['neden']}")
        print(f"      cevap    : {cevap[:110]}")
        kaynaklar = list(dict.fromkeys(src for _, _, _, src in hits))
        print(f"      kaynak   : {', '.join(kaynaklar) if kaynaklar else '(yok — retrieval calismadi)'}")

    print(f"\nUc durum sonucu: {gecen}/{len(UC_DURUMLAR)} gecti "
          f"(bu sayi yukaridaki dogruluk yuzdelerine DAHIL DEGILDIR).")
    return gecen, len(UC_DURUMLAR)


def main() -> None:
    if not config.DB_PATH.exists():
        raise SystemExit("Once calistir:  python ingest.py")

    print("Model yukleniyor...")
    load_chat()
    conn = db.connect()

    dogru_cevap = 0
    isabetli_retrieval = 0
    retrieval_olculen = 0

    print(f"\n{'Soru':<44}{'Cevap':>7}{'Retrieval':>11}")
    print("-" * 62)
    for t in TESTLER:
        cevap, hits = rag.answer(t["soru"], conn=conn)
        kaynaklar = {src for _, _, _, src in hits}

        # 1) Cevap dogrulugu
        cl = sadelestir(cevap)
        cevap_ok = any(sadelestir(a) in cl for a in t["anahtar"])
        dogru_cevap += cevap_ok

        # 2) Retrieval isabeti (sadece bilgi tabaninda OLAN sorular icin)
        if t["kaynak"] is not None:
            retrieval_olculen += 1
            # Dize de liste de kabul edilir; listede kesisim varsa isabet sayilir.
            beklenen = {t["kaynak"]} if isinstance(t["kaynak"], str) else set(t["kaynak"])
            ret_ok = bool(beklenen & kaynaklar)
            isabetli_retrieval += ret_ok
            ret_str = "OK" if ret_ok else "ISKA"
        else:
            ret_str = "-"  # KB'de yok, retrieval beklenmez

        soru_kisa = (t["soru"][:41] + "...") if len(t["soru"]) > 44 else t["soru"]
        print(f"{soru_kisa:<44}{('OK' if cevap_ok else 'YANLIS'):>7}{ret_str:>11}")

    n = len(TESTLER)
    print("-" * 62)
    print(f"\nCevap dogrulugu : {dogru_cevap}/{n}  (%{100*dogru_cevap/n:.0f})")
    print(f"Retrieval isabet: {isabetli_retrieval}/{retrieval_olculen}  "
          f"(%{100*isabetli_retrieval/retrieval_olculen:.0f})")
    print("\nNot: cevap YANLIS ama retrieval OK ise -> uretim (model) sorunu;")
    print("     cevap YANLIS ve retrieval ISKA ise -> retrieval (embedding/top_k) sorunu.")

    uc_durumlari_calistir(conn)
    conn.close()


if __name__ == "__main__":
    main()
