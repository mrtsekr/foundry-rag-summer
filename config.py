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
# Embedding'i Python'da, sentence-transformers ile yapiyoruz; chat/uretim ise
# Foundry Local'da kalir.
# NEDEN: proje basladiginda (2026-07 basi) Foundry Local katalogunda embedding
# modeli YOKTU. 2026-07-31'de tekrar bakildiginda katalogda artik VAR:
#   qwen3-embedding-0.6b (task=embeddings, 495 MB) ve qwen3-embedding-8b;
#   SDK tarafinda model.get_embedding_client() ile kullanilabiliyor.
# Yani bu satirdaki tercih artik bir zorunluluk degil, bir SECIM: mevcut kurulum
# olculmus ve calisiyor. Foundry embedding'e gecmek acik bir iyilestirme yolu
# (bkz. README > Sonraki adimlar) ama EMBED_DIM ve yeniden ingest gerektirir.
# Embedding modeli gecmisi:
#   1) all-MiniLM-L6-v2 (Ingilizce-merkezli) -> Turkce'de zayif retrieval
#   2) paraphrase-multilingual-MiniLM-L12-v2 -> daha iyi ama dolayli sorularda iskaliyordu
#   3) multilingual-e5-small -> retrieval icin ozel egitilmis, dolayli/parafraz
#      sorularda belirgin daha iyi. Yine 384 boyut (sema degismedi).
# DIKKAT: e5 modelleri metne ON-EK ister -> sorgu icin "query: ", belge parcasi
# icin "passage: ". Bu on-ekler embed_texts() icinde otomatik eklenir.
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_DIM = 384  # e5-small'in urettigi vektor boyutu
EMBED_QUERY_PREFIX = "query: "      # sorulari kodlarken bu on-ek eklenir
EMBED_PASSAGE_PREFIX = "passage: "  # belge parcalarini kodlarken bu on-ek eklenir

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
