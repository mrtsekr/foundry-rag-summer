"""
Merkezi ayar dosyasi.
Butun sabitler (model adlari, veritabani yolu, chunk boyutu vb.) burada tutulur.
Boylece bir seyi degistirmek istedigimizde tek yere bakariz.

Ornek: chat modelini degistirmek icin sadece CHAT_MODEL satirini duzenle.
"""

from pathlib import Path

# --- Proje koku (bu dosyanin bulundugu klasor) ---
PROJECT_DIR = Path(__file__).resolve().parent

# --- Embedding (retrieval) modeli ---
# IKI arka uc destekleniyor; asagidaki tek satirla secilir:
#
#   "st"      -> multilingual-e5-small, sentence-transformers (VARSAYILAN)
#   "foundry" -> qwen3-embedding-0.6b, Foundry Local SDK
#
# NEDEN "st" VARSAYILAN — OLCULDU (bench_embed.py, uretim ayarlari, 9 parca /
# 16 soru). Retrieval deterministiktir, yani bu sayilar tur tur oynamaz:
#
#                        e5-small(384)   qwen3-emb(1024)
#   kanit ilk 3'te ***    15/16 (%94)     14/16 (%88)     <- KARAR SATIRI
#   dogru dosya ilk 3te   14/16 (%88)     13/16 (%81)
#   kanit ilk 1'de        12/16 (%75)     13/16 (%81)
#   MRR                        0.840           0.854
#   16 soruyu embedleme      0.08 sn        11.75 sn
#
# TOP_K=3 oldugu icin "ilk 3'te mi" karar satiridir: modelin gercekten GORDUGU
# bilgi budur. qwen ilk-1 ve MRR'de bir tik onde ama bu ustunluk ilk 3'un
# icinde kaliyor -> cevaba yansimiyor. Buna karsilik "Ucaktan indim, otele
# nasil ulasabilirim?" sorusunda dogru parcayi 2. siradan 5.'ye dusuruyor:
# ilk-3'un DISINA cikiyor ve uctan uca eval'de o soru iskaya donusuyor.
# Ustelik sorgu basina ~0,73 sn ekliyor (cevap suresi ~2,5 sn iken %30 fazladan
# gecikme), 20x daha buyuk ve GPU/VRAM ile yarisiyor.
#
# Yani plan (s.7) qwen3-embedding-0.6b'yi oneriyor olsa da, bu proje icin
# dogru secim olculen sonuca gore e5-small. Foundry yolu SILINMEDI: calisir,
# olculmus ve tek satirla acilabilir halde duruyor (ayni eval ile tekrar
# kiyaslanabilsin diye).
#
# ARKA UCU DEGISTIRINCE "python ingest.py" TEKRAR CALISTIRILMALIDIR: vektor
# boyutu (384 <-> 1024) ve vektor uzayi degisir, eski veritabani gecersiz olur.
# db.search() bunu yakalar ve acik bir hata mesaji verir.
#
# Embedding modeli gecmisi (hepsi olculdu):
#   1) all-MiniLM-L6-v2 (Ingilizce-merkezli) -> Turkce'de zayif retrieval
#   2) paraphrase-multilingual-MiniLM-L12-v2 -> daha iyi ama dolayli sorularda iskaliyordu
#   3) multilingual-e5-small -> retrieval icin ozel egitilmis, 384 boyut  <- SECILEN
#   4) qwen3-embedding-0.6b (Foundry Local) -> 1024 boyut, olculdu, kazanamadi
EMBED_BACKEND = "st"   # "st" | "foundry"

# -- Arka uc: Foundry Local --
# 495 MB'lik generic-cpu varyanti; sohbet modeli (qwen3-4b) GPU'da oldugu icin
# embedding'in CPU'da kalmasi VRAM acisindan da isimize geliyor.
FOUNDRY_EMBED_MODEL = "qwen3-embedding-0.6b"
FOUNDRY_EMBED_DIM = 1024
# Qwen3-Embedding SORGULAR icin gorev tanimli bir on-ek onerir; belgeler HAM
# verilir. Olcumde ham sorgu 11/16, instruct'li sorgu 13/16 isabet vermisti.
FOUNDRY_QUERY_INSTRUCT = (
    "Instruct: Given a Turkish question about a hotel, retrieve the passage that answers it\n"
    "Query: "
)

# -- Arka uc: sentence-transformers --
# DIKKAT: e5 modelleri metne ON-EK ister -> sorgu "query: ", parca "passage: ".
ST_EMBED_MODEL = "intfloat/multilingual-e5-small"
ST_EMBED_DIM = 384
ST_QUERY_PREFIX = "query: "
ST_PASSAGE_PREFIX = "passage: "

# -- Secilen arka ucun turetilmis degerleri (kodun geri kalani bunlari okur) --
EMBED_MODEL = FOUNDRY_EMBED_MODEL if EMBED_BACKEND == "foundry" else ST_EMBED_MODEL
EMBED_DIM = FOUNDRY_EMBED_DIM if EMBED_BACKEND == "foundry" else ST_EMBED_DIM

# --- Chat / uretim (generation) modeli — Foundry Local uzerinde calisir ---
# Donanim: RTX 3060 Laptop (6 GB VRAM). qwen3-4b (~2.63 GB) VRAM'e rahat sigar,
# Turkce'de iyi, GPU'da hizli. Turkce kalitesi yetmezse "qwen3.5-4b" yap (daha
# yeni, daha iyi Turkce ama VRAM'e daha sikca oturur).
CHAT_MODEL = "qwen3-4b"

# --- Bilgi tabani (kaynak belgeler) ---
# ingest.py bu klasordeki tum .txt dosyalarini okuyup parcalar ve veritabanina yazar.
DOCS_DIR = PROJECT_DIR / "docs"

# --- SQLite veritabani ---
# Belge parcalarini (chunk) ve onlarin embedding'lerini kalici olarak saklariz;
# boylece her acilista yeniden hesaplamaya gerek kalmaz.
DB_PATH = PROJECT_DIR / "rag_store.db"

# --- Chunk (parcalama) ayarlari ---
# Uzun belgeleri retrieval icin kucuk parcalara boleriz. overlap, iki parca
# arasinda bir miktar ortusme birakir ki cumleler tam ortadan bolunup anlam
# kaybolmasin.
# Paragraf-oncelikli parcalama: her paragraf (konu) mumkun oldugunca kendi
# parcasina duser -> tek bir konu (or. spa/masaj) baska konularin arasinda
# gomulup embedding'i seyrelmez, retrieval'da daha iyi bulunur.
# OLCULDU (2026-07-31): 350'de her paragraf ayri parcaya dusuyordu (31 parca) ve
# retrieval bundan ZARAR goruyordu — bir konunun cevabi komsu paragraftaki
# baglamdan kopuyordu. Paragraflari 1000 karaktere kadar TOPARLAYINCA bu
# projedeki belgeler (ort. 718 krk) tek parca kaliyor ve cevap dogrulugu
# %76-82'den %88-100'e cikti (3'er tur eval).
# 1000 keyfi degil: kisa belgeleri bolmeyecek kadar buyuk, uzun bir belgeyi
# planin onerdigi "1-3 paragrafl" pasajlara bolecek kadar kucuk. Yani ayni kod
# sayfalarca belgede de dogru davranir (o zaman tek belge birden cok parca olur).
CHUNK_MAX_CHARS = 1000  # bir parcanin hedef/ust sinir uzunlugu
CHUNK_MIN_CHARS = 120   # sona kalan bu kadar kisa parca tek basina birakilmaz
CHUNK_OVERLAP = 80      # tek basina max'tan uzun paragraf bolunurken ortusme

# --- Retrieval ayari ---
# Bir soruya cevap verirken en benzer kac parcayi modele baglam olarak verecegiz.
TOP_K = 3
