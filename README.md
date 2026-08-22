# Deniz Yıldızı Resort — Offline Türkçe RAG Asistanı

Tamamen yerel çalışan Türkçe bir soru-cevap (RAG) asistanı. Bir otelin
belgelerini okur ve misafir sorularını yalnızca o belgelere dayanarak yanıtlar.
Bulut yok, API anahtarı yok, veri cihazdan çıkmaz.

Bir aylık yaz okulu projesi. Amacı RAG'in her parçasını (parçalama, embedding,
retrieval, üretim) hazır bir çerçeveye sarılmadan elle kurmak ve her ayarı
ölçerek seçmek.

**Tanıtım sayfası:** <https://mrtsekr.github.io/foundry-rag-summer/>
Mimari, ölçümler, alınan ve reddedilen kararlar, 21 gerçek çalıştırmada arama.
Tek HTML dosyası, sunucu istemez. Kendi bilgisayarında `python site/sunucu.py`
ile açarsan sayfa canlı moda geçer ve istediğin soruyu gerçekten modele sorarsın.

| | |
|---|---|
| Üretim (chat) | [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/) + `qwen3-4b`, yerel GPU'da (RTX 3060 Laptop, 6 GB) |
| Retrieval | `sentence-transformers` + `intfloat/multilingual-e5-small` (384 boyut) |
| Depo | Python'un yerleşik `sqlite3`'ü, ayrı vektör veritabanı yok |
| Bilgi tabanı | 9 belge → 9 parça (`.txt` · `.md` · `.pdf`) |

---

## Kurulum

Python 3.12 gerekiyor (torch ve ML kütüphaneleri için 3.14 henüz çok yeni).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Foundry Local kurulu ve servisi ayakta olmalı (`foundry service status`).

İlk çalıştırmada iki tek seferlik maliyet var: program GPU çalışma
sağlayıcılarını bu sürece kaydeder (10-15 dakika sürebilir, `llm.py` içindeki
`download_and_register_eps`), ve embedding modeli indirilir (~120 MB).
Sonrasında sistem tamamen çevrimdışı çalışır; `HF_HUB_OFFLINE=1` ve
`TRANSFORMERS_OFFLINE=1` ile doğrulandı.

## Kullanım

```bash
python ingest.py        # bilgi tabanını kur (docs/ altındaki .txt/.md/.pdf)
python assistant.py     # etkileşimli asistan, her cevapta kaynak belge
```

Ölçüm araçları:

```bash
python eval.py                  # cevap doğruluğu + retrieval isabeti + uç durumlar
python bench/tune_topk.py       # TOP_K taraması (yalnız retrieval, hızlı)
python bench/bench_embed.py     # embedding arka uçlarını kıyasla (LLM çalışmaz)
python bench/bench_chunking.py  # parçalama × embedding taraması
```

Kendi belgelerini eklemek için `docs/` klasörüne `.txt`, `.md` veya `.pdf` koy,
`python ingest.py` komutunu tekrar çalıştır.

---

## Mimari

```
  docs/*.txt, *.md, *.pdf
          │
          ▼   ingest.py       paragraf-öncelikli parçalama + embedding
   ┌─────────────────┐
   │  rag_store.db   │        SQLite: documents + chunks(text, embedding BLOB)
   └─────────────────┘
          │
   SORU ──┤
          ▼   db.search()     soruyu embed'le, kosinüs benzerliği, en iyi TOP_K parça
          ▼   rag.answer()    parçaları "BAĞLAM" olarak prompt'a yerleştir
          ▼   llm.generate()  qwen3-4b (Foundry Local, GPU) → Türkçe cevap
          ▼
        CEVAP  (+ kaynak belge adı)
```

Model ezberinden değil, getirilen belgeden cevaplar. Belgede yoksa uydurmaz,
"Bu konuda elimde bilgi yok, resepsiyona danışabilirsiniz." der.

## Dosya yapısı

| Yol | Görev |
|-----|-------|
| `config.py` | Tüm ayarlar tek yerde: model adları, DB yolu, parça boyutları, `TOP_K` |
| `ingest.py` | Belge okuma → `chunk_text` → embedding → SQLite |
| `db.py` | SQLite şeması, embedding BLOB dönüşümü, `search()` |
| `rag_core.py` | `embed_texts()` (seçili arka uca göre) ve `cosine_similarity()` |
| `foundry.py` | Foundry Local'a tek giriş: manager, EP kaydı, GPU varyant seçimi |
| `llm.py` | `qwen3-4b`'yi GPU'da yükler, üretir, tekrarı kırpar |
| `rag.py` | Uçtan uca RAG: retrieval → bağlam → üretim |
| `assistant.py` | Etkileşimli komut satırı asistanı |
| `eval.py` | 17 soruluk değerlendirme (cevap ve retrieval ayrı) + 5 uç durum testi |
| `docs/` | Bilgi tabanı: 9 belge, Türkçe otel içeriği |
| `bench/` | Ölçüm araçları: `tune_topk.py`, `bench_embed.py`, `bench_chunking.py` |
| `tani/` | Tanı scriptleri: `compare_chat.py`, `diag_variants.py`, `diag_eps.py` |
| `site/` | Tanıtım sayfası: `index.html`, `kayitlar.json`, `kayit_uret.py`, `sunucu.py` |

`tani/` altındakiler tek seferlik teşhis scriptleri ama duruyorlar, çünkü sitede
yayımlanan VRAM ve süre sayılarını bunlar üretti. Silinirse ölçümlerin kaynağı
kaybolur.

---

## Teknik kararlar

Bu bölüm özet. Ölçümlerin tamamı, grafikleri ve reddedilen denemeler
[tanıtım sayfasında](https://mrtsekr.github.io/foundry-rag-summer/).

**Chat modeli `qwen3-4b`, çünkü kart öyle istedi.** Aday `qwen3.5-4b` 6 GB'lık
karta sığmadı: tepe kullanım 5903/6144 MiB'e tırmandı, paylaşımlı belleğe taştı,
cevap süresi 115 saniyeye çıktı. `qwen3-4b` 4516 MiB'de kalıyor ve 2,5 saniyede
cevaplıyor. Ölçüm `tani/compare_chat.py`.

**GPU'yu elle seçmek gerekiyor.** Foundry SDK, çalışma sağlayıcıları bu sürece
kayıtlı değilken sessizce CPU varyantını seçiyor. Model doğru çalışır ama sürünür
(21,6 sn/cevap) ve hata mesajı vermez. `llm.py` önce EP'leri kaydedip sonra
`cuda-gpu` varyantını açıkça seçiyor: 21,6 sn → 2,5 sn.

**Embedding çok dilli olmalı.** İngilizce merkezli `all-MiniLM-L6-v2` Türkçe
sorularda yanlış parça getirdi ve model uydurdu. Retrieval için özel eğitilmiş
`multilingual-e5-small` en iyi sonucu verdi. Üçü de 384 boyut olduğu için şema
hiç değişmedi. e5 modelleri ön ek ister: sorguya `query: `, parçaya `passage: `.

**`temperature = 0.7`, sezgiye ters ama doğru.** Model aynı cümleyi 20 kez
tekrarlıyordu. Suçlu sanılan sıcaklık aslında çözümdü: düşük sıcaklıkta (0.3)
üretim aynı token dizisine saplanıyordu. 0.7 döngüyü kırdı, cevap bağlama dayalı
olduğu için kalite bozulmadı.

**`frequency_penalty` kullanılmadı.** Tekrar sorununa bariz çözüm gibi görünüyordu
ama işi kötüleştirdi: cevapları boşalttı, bir soruda "kumarhane yok"u "var"a
çevirerek tehlikeli bir halüsinasyon üretti. Doğru araç sıcaklıktı.

**Parçaları bölmemek.** Başlangıçta her paragraf ayrı parçaya düşüyordu (31 parça)
ve sezgi "küçük parça = keskin embedding" diyordu. Ölçüm bunu çürüttü: bir konunun
cevabı komşu paragraftaki bağlamdan koparılıyordu. Parçaları 1000 karaktere kadar
toparlayınca 31 parça 9'a indi ve cevap doğruluğu %80'den %88'e çıktı.

**Foundry'nin embedding modeline geçilmedi.** Proje başladığında Foundry Local
kataloğunda embedding modeli yoktu; retrieval bu yüzden Python tarafında kuruldu.
Temmuz sonunda katalog yeniden kontrol edildiğinde `qwen3-embedding-0.6b` çıkmıştı.
Yani mevcut kurulum artık zorunluluk değil, tercih. Tercih ölçülerek verildi:

| Ölçüt (16 soru, 9 parça) | e5-small (384) | qwen3-emb (1024) |
|---|---|---|
| Kanıt ilk 3'te (karar metriği) | **15/16 · %94** | 14/16 · %88 |
| Doğru dosya ilk 3'te | **14/16 · %88** | 13/16 · %81 |
| MRR | 0.840 | **0.854** |
| 16 soruyu embedleme | **0,08 sn** | 11,75 sn |

`TOP_K = 3` olduğu için karar satırı "kanıt ilk 3'te": modelin gerçekten gördüğü
bilgi budur. 20 kat büyük model orada geride kaldı ve sorgu başına ~0,73 saniye
ekledi. Foundry yolu silinmedi, `config.EMBED_BACKEND` tek satırda geri açıyor.

İlk kıyas taraflıydı, çünkü parçalar e5 için optimize edilmiş bir ayarla
kurulmuştu. `bench/bench_chunking.py` aynı iki modeli dört farklı parçalama
ayarında ölçüyor; küçük parçalarda Qwen öne geçiyor, bu korpusun gerçek
ayarında (9 parça) fark kapanıyor.

**`TOP_K = 3`.** `bench/tune_topk.py` taramasında isabet k=3'te doyuyor, daha
büyük k üretime yalnızca gürültü taşıyor.

---

## Değerlendirme

`python eval.py` iki metriği ayrı ölçer: cevap doğruluğu ve retrieval isabeti.
Ayırmak, bir hatanın nerede olduğunu söyler. Yanlış parça mı geldi, yoksa doğru
parça gelip model mi yanlış cevapladı?

| Metrik | Sonuç |
|---|---|
| Cevap doğruluğu (17 soru, 4 çalıştırma) | 13-16 / 17, ortalama %84 |
| Retrieval isabeti (16 soru, deterministik) | 15 / 16, %94 |
| Uç durum testleri | 5 / 5 |
| Ortalama cevap süresi | 2,5 sn |

Cevap doğruluğu tek sayı değil aralık, çünkü üretim örneklemeli
(`temperature = 0.7`) ve aynı soru çalıştırmalar arasında farklı cevaplanabiliyor.
En iyi tur seçilmedi, gözlenen aralık yazıldı. Retrieval deterministik olduğu için
hiç oynamıyor.

Uç durum testleri ana yüzdelere karıştırılmadı. Ana set "doğru cevabı biliyor mu",
uç durumlar "beklenmedik girdide güvenli mi" sorusunu ölçüyor; beşi de kolay
geçtiği için aynı tabloya konsaydı doğruluk suni biçimde şişerdi.

### Güvenli başarısızlık

Kalan hatalar bilerek güvenli tarafta bırakıldı: sistem bilmediğinde uydurmuyor,
"bilgim yok" diyor. Bunun bedeli birkaç puan doğruluk, karşılığı misafire yanlış
bilgi vermemek.

Bu davranışı `rag.py` içindeki tek bir few-shot örneği pekiştiriyor ve o örnek
bilerek olgu taşımıyor. Sebebi bir deneyle öğrenildi: örneğe somut bir sayı
konduğunda ("200 oda"), model doğru parçayı getiremediği bir soruda o sayıyı
cevaba sızdırdı. Yani few-shot örneğinin kendisi halüsinasyon kaynağına dönüştü.
Örnek veri taşımayan bir güvenlik örneğine indirildi; doğruluk aynı kaldı,
sızıntı bitti.

---

## Bilinen sınırlamalar

- **Retrieval metriği fazla katı.** Her soru için tek bir doğru kaynak bekliyor.
  "Otelde casino var mı" sorusunda beklenen `aktiviteler.txt` yerine
  `otel_genel.txt` geliyor; o belge de casino'dan bahsettiği için cevap doğru
  çıkıyor ama metrik ıska sayıyor. %94 bu yüzden kötümser.
- **Benzerlik skorları birbirine çok yakın.** Bütün skorlar 0,77-0,87 aralığına
  sıkışıyor, yani yüksek skor tek başına "doğru parça geldi" demek değil.
  Parçaları büyütmek bunu hafifletti, kalan pay için bir yeniden sıralama adımı
  gerekir.
- **"Belge = parça" yaklaşımı bu korpusa özel.** Sayfalarca belgeye ölçeklenmez:
  uzun bir belgenin tek embedding'i tüm konuların ortalaması olur. Parçalayıcı
  bunu ele alıyor (1000 karakteri aşan belgeler bölünüyor) ama o rejim bu veriyle
  sınanmadı. Ayrıca `db.search()` tüm vektörleri belleğe alıp tek tek
  karşılaştırıyor; büyük N için gerçek bir vektör indeksi gerekir.
- **Model tavanı.** `qwen3-4b` 6 GB'a sığan en iyi seçenek ama günlük ve dolaylı
  ifadelerde ara sıra yanlış cümleye takılıyor. `llm.py` içindeki tekrar kırpıcı
  döngüyü keser, içerik hatasını düzeltmez.
- **PDF yalnızca metin tabanlı.** Taranmış (görüntü) PDF'ten metin çıkmaz, OCR yok.
- **Değerlendirme seti küçük** (17 soru) ve elle hazırlandı. İstatistiksel güven
  aralığı iddia etmiyor, yön gösteriyor.
- **Oturum hafızası yok.** Her soru bağımsız; çok turlu diyalog kurulmadı.

## Sonraki adımlar

- **Embedding kıyasını daha büyük bir bilgi tabanında tekrarlamak.** Bugünkü 9
  parçalık korpusta e5-small önde çıktı, ama `bench/bench_chunking.py` Qwen'in
  parça sayısı arttıkça güçlendiğini gösteriyor. Eksik olan kod değil veri: arka
  uç kurulu, `config.EMBED_BACKEND` tek satır. Yeni belgeler gerçek olduğunda
  anlamlı olur; sırf ölçüm yapmak için sentetik belge üretmek hem sonucu hem
  anlatıyı bozar.
- **Yeniden sıralama (reranker).** Bu ölçekte işe yaramaz: toplam 9 parça var ve
  `TOP_K = 3`. Reranker'ın işi çok sayıda aday arasından iyi olanı yukarı çekmek,
  9 adayın olduğu yerde sıralanacak bir şey yok. Üstelik ikinci bir model cevap
  süresine eklenir. Parça sayısı birkaç yüze çıkarsa gündeme gelir.
- **Çok turlu diyalog.** Şu an her soru bağımsız. Önceki soruya atıf yapan
  ("peki orada kahvaltı kaçta?") sorular için soru yeniden yazma adımı gerekir.
  Değerlendirme seti de buna göre genişletilmeli, yoksa iyileşme ölçülemez.

---

## Veri hakkında

Bilgi tabanındaki belgeler, Kuzey Kıbrıs'ta faaliyet gösteren gerçek bir resort
otelin herkese açık web sitesindeki bilgilerden uyarlandı. Otelin ve alt
markalarının adları anonimleştirildi: tesis bu depoda "Deniz Yıldızı Resort &
Casino" olarak geçiyor, restoran ve bar marka adları tür adlarına çevrildi
(örneğin "İtalyan à la carte restoranı"). Oda sayısı, mesafe, saat gibi olgusal
veriler korundu ki sistem gerçekçi bir veri üzerinde ölçülebilsin.

Bu depo eğitim ve gösterim amaçlıdır. Herhangi bir otelle ticari bağı yoktur,
resmî bir bilgi kaynağı değildir ve otel adına bilgi vermez.

## Kaynaklar

Kullanılan araçlar ve modeller:

- [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/)
  ve [foundry-local-sdk](https://pypi.org/project/foundry-local-sdk/) — yerel model çalıştırma
- [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) — üretim modeli, Foundry Local kataloğundan
- [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) — embedding modeli
- [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — kıyasta denenen alternatif
- [sentence-transformers](https://www.sbert.net/) — embedding arayüzü
- [pypdf](https://pypi.org/project/pypdf/) — PDF metin çıkarma

Yöntem için başvurulanlar:

- Lewis ve ark., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401) (2020) — RAG'in özgün tanımı
- Wang ve ark., [*Multilingual E5 Text Embeddings*](https://arxiv.org/abs/2402.05672) (2024) — e5 ön ek kuralı (`query:` / `passage:`) buradan
- [Foundry Local SDK dokümantasyonu](https://learn.microsoft.com/azure/ai-foundry/foundry-local/reference/reference-sdk) — EP kaydı ve varyant seçimi

Tanıtım sayfasındaki bileşenler [21st.dev](https://21st.dev) kaynaklarından
uyarlandı (Timeline, Hero Highlight, AI Agent Pipeline, animated-beam,
bento-grid). Hepsi React bileşeniydi; bu projede düz HTML/CSS'e taşındı.

## Lisans

Kod [MIT lisansı](LICENSE) altındadır.

Bu lisans `docs/` klasörünü kapsamaz. Oradaki belgeler üçüncü bir tarafın herkese
açık içeriğinden uyarlanmış ve anonimleştirilmiş örnek verilerdir; yalnızca
gösterim amacıyla bulunuyorlar.
