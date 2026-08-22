"""
Hafta 3 · Uctan uca RAG — ilk gercek soru-cevap!
Akis:  soru -> (retrieval) en benzer parcalar -> (generation) qwen3-4b Turkce cevap

  1) db.search(): soruyu embed'ler, bilgi tabanindan en benzer TOP_K parcayi getirir
  2) Bu parcalari 'BAGLAM' olarak bir prompt'a koyariz
  3) llm.generate(): qwen3-4b SADECE bu baglama dayanarak Turkce cevap uretir

Once bilgi tabanini kur (bir kez):  python ingest.py
Sonra:                              python rag.py
"""

import config
import db
from llm import generate

SYSTEM = (
    "Sen Deniz Yildizi Resort & Casino'nun yardimci musteri asistanisin. "
    "SADECE sana verilen BAGLAM'daki bilgilere dayanarak cevap ver. "
    "Cevabini en fazla 2 cumlede, dogal bir Turkce ile ver; kendini TEKRAR ETME. "
    "Baglamda cevap yoksa 'Bu konuda elimde bilgi yok, resepsiyona danisabilirsiniz.' de. "
    "Baglamda olmayan bilgiyi asla uydurma."
)

# Few-shot ornegi: SADECE bir "guvenlik" ornegi tutuyoruz -> baglamda cevap YOKSA
# uydurma, "bilgi yok" de.
# NEDEN sadece bu? Ornege somut bir olgu (sayi/isim) koyarsak, model bilmedigi bir
# soruda o olguyu SIZDIRABILIR (deneyde "200 oda" ornegi, 569 getirilemeyince
# cevaba sizdi). Bu ornek olgu tasimadigi icin guvenlidir: yalnizca "uydurma"
# davranisini pekistirir, cevaba veri sizdirmaz.
EXAMPLES = [
    ("BAGLAM:\n- Otelde acik ve kapali havuzlar bulunur.\n\nSORU: Otelde bowling salonu var mi?",
     "Bu konuda elimde bilgi yok, resepsiyona danisabilirsiniz."),
]


# Bos (ya da yalnizca bosluk olan) sorgu icin sabit yanit.
# NEDEN kapida durduruyoruz: bos metin de embed'lenebilir -> db.search yine de
# TOP_K parca dondurur (skorlari anlamsizdir) ve model bu rastgele baglamla
# "bir seyler" yazar; yani sistem sessizce alakasiz bir cevap uydurur.
# Girdiyi burada kesmek hem dogru davranis hem de bedava: GPU'ya hic gidilmez.
BOS_SORU_MESAJI = (
    "Bir soru yazmadiniz. Ornegin 'Kahvalti kacta baslar?' diye sorabilirsiniz."
)


def answer(question: str, conn=None, goster=False):
    """Bir soruya RAG ile cevap uretir. (cevap, kaynaklar) dondurur.

    Bos/bosluk-only sorguda retrieval ve uretim hic calistirilmaz: sabit bir
    yonlendirme mesaji ve BOS kaynak listesi doner.
    """
    question = (question or "").strip()
    if not question:
        return BOS_SORU_MESAJI, []

    kapat = False
    if conn is None:
        conn = db.connect()
        kapat = True

    # 1) Retrieval
    hits = db.search(conn, question, config.TOP_K)

    # 2) Baglami olustur
    baglam = "\n\n".join(f"- {text}" for _, _, text, _ in hits)
    user = f"BAGLAM:\n{baglam}\n\nSORU: {question}"

    if goster:
        print("  [getirilen parcalar]")
        for skor, _, text, source in hits:
            ozet = text.replace("\n", " ")[:70]
            print(f"    {skor:.3f} [{source}] {ozet}...")

    # 3) Generation (few-shot ornekleriyle)
    cevap = generate(SYSTEM, user, examples=EXAMPLES)

    if kapat:
        conn.close()
    return cevap, hits


def _demo():
    if not config.DB_PATH.exists():
        raise SystemExit("Once bilgi tabanini kur: python ingest.py")

    sorular = [
        "Kahvaltiyi en gec kacta yiyebilirim?",
        "Havuz kacta kapaniyor?",
        "Odaya evcil hayvan getirebilir miyim?",
        "Check-out saati nedir, gec cikis mumkun mu?",
        "Otelde spor salonu var mi?",
        "Odalarda kasa var mi?",
        "Otelde kumarhane var mi?",  # baglamda YOK -> 'bilgim yok' demeli
    ]

    conn = db.connect()
    print("Model yukleniyor (ilk cevap oncesi ~10-15 sn)...\n")
    for soru in sorular:
        print(f"SORU: {soru}")
        cevap, _ = answer(soru, conn=conn, goster=True)
        print(f"CEVAP: {cevap}\n{'-'*60}")
    conn.close()


if __name__ == "__main__":
    _demo()
