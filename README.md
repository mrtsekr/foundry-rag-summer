# Deniz Yıldızı Resort — Offline Türkçe RAG Asistanı

Tamamen **yerel ve offline** çalışan Türkçe bir soru-cevap (RAG) asistanı.
Bir otelin belgelerini (odalar, yeme-içme, spa, politikalar, ulaşım…) okur ve
misafir sorularını **yalnızca o belgelere dayanarak** yanıtlar. Bulut yok, API
anahtarı yok, veri cihazdan çıkmaz.

| | |
|---|---|
| **Üretim (chat)** | [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/) + `qwen3-4b`, yerel GPU'da (RTX 3060 Laptop, 6 GB) |
| **Retrieval (embedding)** | `sentence-transformers` + `intfloat/multilingual-e5-small` (384 boyut) |
| **Depo** | Python'un yerleşik `sqlite3`'ü — ayrı vektör veritabanı sunucusu yok |
| **Bilgi tabanı** | 9 belge → 9 parça (`.txt` · `.md` · `.pdf`) |

> Bir aylık yaz projesi. Amaç: RAG'in her parçasını — parçalama → embedding →
> retrieval → üretim — hazır bir çerçeveye sarılmadan elle kurup ölçerek
> öğrenmek. Bu depodaki her karar bir ölçümle verildi; ölçümler aşağıda,
> **cilalanmamış haliyle** paylaşılıyor.

---

## Veri hakkında (atıf)

Bilgi tabanındaki belgeler, Kuzey Kıbrıs'ta faaliyet gösteren gerçek bir resort
otelin **herkese açık web sitesindeki** bilgilerden uyarlanmıştır. Otelin ve
alt markalarının adları **anonimleştirilmiştir**: tesis bu depoda "Deniz Yıldızı
Resort & Casino" olarak geçer, restoran/bar marka adları ise tür adlarına
(ör. "İtalyan à la carte restoranı") çevrilmiştir. Oda sayısı, mesafe, saat gibi
**olgusal veriler korunmuştur** ki sistem gerçekçi bir veri üzerinde ölçülebilsin.

Bu depo **eğitim ve gösterim amaçlıdır**; herhangi bir otelle ticari bağı yoktur,
resmî bir bilgi kaynağı değildir ve otel adına bilgi vermez.

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
          ▼   db.search()     soruyu embed'le → kosinüs benzerliği → en iyi TOP_K parça
          ▼   rag.answer()    parçaları "BAĞLAM" olarak prompt'a yerleştir
          ▼   llm.generate()  qwen3-4b (Foundry Local, GPU) → Türkçe cevap
          ▼
        CEVAP  (+ kaynak belge adı)
```

**RAG = Retrieval + Generation.** Model ezberinden değil, getirilen belgeden
cevaplar. Belgede yoksa uydurmaz, "Bu konuda elimde bilgi yok, resepsiyona
danışabilirsiniz." der — bkz. [Güvenli başarısızlık](#güvenli-başarısızlık).

---

## Kurulum

1. **Python 3.12** (torch ve ML kütüphaneleri için 3.14 henüz çok yeni).
2. Sanal ortam + paketler:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Foundry Local** kurulu ve servisi ayakta olmalı (`foundry service status`).
4. **İlk çalıştırma notu:** Program, GPU çalışma sağlayıcılarını (CUDA/TensorRT)
   ilk kez bu sürece kaydeder — **tek seferlik ~10-15 dakika** sürebilir; sonraki
   çalıştırmalar hızlıdır. (`llm.py` → `download_and_register_eps`.)

Embedding modeli ilk çalıştırmada bir kez indirilir (~120 MB). Sonrasında sistem
tamamen offline çalışır; `HF_HUB_OFFLINE=1` ve `TRANSFORMERS_OFFLINE=1` ile
doğrulanmıştır.

## Kullanım

```bash
# 1) Bilgi tabanını kur (docs/ altındaki tüm .txt/.md/.pdf dosyalarını işler)
python ingest.py

# 2) Etkileşimli asistanı başlat (sürekli soru-cevap, her cevapta kaynak belge)
python assistant.py

# 3) (opsiyonel) Kaliteyi ölç
python eval.py          # cevap doğruluğu + retrieval isabeti + uç durumlar
python tune_topk.py     # TOP_K taraması (yalnız retrieval, hızlı)
python bench_embed.py   # embedding arka uçlarını kıyasla (LLM çalışmaz)
```

Kendi belgelerini eklemek için `docs/` klasörüne `.txt`, `.md` veya `.pdf` koy ve
`python ingest.py` komutunu tekrar çalıştır.

---

## Dosya yapısı

| Dosya | Görev |
|-------|-------|
| `config.py` | Tüm ayarlar tek yerde (model adları, DB yolu, chunk boyutları, `TOP_K`) |
| `docs/` | Bilgi tabanı belgeleri (9 belge, Türkçe otel içeriği) |
| `ingest.py` | Belge okuma (`.txt`/`.md`/`.pdf`) → `chunk_text` → embedding → SQLite |
| `db.py` | SQLite şeması, embedding BLOB dönüşümü, `search()` (retrieval) |
| `rag_core.py` | `embed_texts()` — seçili arka uca göre (e5 ön-ekleri ya da Foundry) + `cosine_similarity()` |
| `foundry.py` | Foundry Local'a tek giriş: paylaşılan manager, EP kaydı, GPU varyant seçimi |
| `llm.py` | `qwen3-4b`'yi GPU'da yükler, üretir, tekrarı kırpar |
| `rag.py` | Uçtan uca RAG: retrieval → bağlam → üretim (system prompt + few-shot) |
| `assistant.py` | Etkileşimli komut satırı asistanı |
| `eval.py` | 17 soruluk değerlendirme (cevap **ve** retrieval ayrı) + 5 uç durum testi |
| `tune_topk.py` | Retrieval-only `TOP_K` taraması |
| `bench_embed.py` | Embedding arka uçlarının kıyası (LLM'siz, deterministik): e5-small vs `qwen3-embedding-0.6b` |
| `bench_chunking.py` | Parçalama × embedding taraması (kıyasın kendi ayarımıza taraflılığını ölçer) |

Öğrenme/tanı scriptleri: `hello_model.py` (Foundry testi), `embed_test.py`
(embedding testi), `compare_chat.py` (model karşılaştırması), `diag_*.py`.

---

## Önemli teknik kararlar (ve nedenleri)

**Chat modeli = `qwen3-4b`.** Aday `qwen3.5-4b` (5.36 GB) 6 GB'lık karta sığmadı:
VRAM 5903/6144 MiB'e tırmandı, paylaşımlı belleğe taştı, cevap başına ~115 sn.
`qwen3-4b` peak 4516/6144 MiB'de kalıyor ve cevap başına ~2.5 sn. Donanım kararı verdi.

**GPU'yu elle seçmek gerekiyor.** Foundry SDK, çalışma sağlayıcıları (EP) bu
sürece kayıtlı değilken sessizce `-generic-cpu` varyantını seçiyor — model doğru
çalışır ama sürünür (21.6 sn/cevap). `llm.py` önce EP'leri kaydeder, sonra
`cuda-gpu` varyantını **açıkça** seçer: 21.6 sn → 2.5 sn.

**Embedding çok-dilli olmalı — ölçüldü.** İngilizce merkezli `all-MiniLM-L6-v2`
Türkçe sorularda (havuz, evcil hayvan) yanlış parça getirdi ve model uydurdu.
`paraphrase-multilingual-MiniLM-L12-v2` bunu düzeltti; retrieval için özel
eğitilmiş `multilingual-e5-small` ise dolaylı sorularda daha da iyi oldu
(retrieval isabeti %88 → %94). Üçü de 384 boyut, yani şema hiç değişmedi.
e5 modelleri ön-ek ister: sorguya `query: `, parçaya `passage: ` (`rag_core.py`).

**`temperature = 0.7` — sezgiye ters ama doğru.** Model aynı cümleyi 20 kez
tekrarlıyordu; suçlu sanılan sıcaklık aslında çözümdü: düşük sıcaklıkta (0.3)
greedy üretim aynı token dizisine saplanıyordu. 0.7 döngüyü kırdı ve cevap
bağlama dayalı olduğu için kaliteyi bozmadı.

**`frequency_penalty` KULLANILMADI.** Tekrar sorununa "bariz" çözüm olarak
denendi ve işi kötüleştirdi: cevapları boşalttı, hatta "kumarhane **yok**"u
"**var**"a çevirerek tehlikeli bir halüsinasyon üretti. Doğru araç sıcaklıktı.

**Embedding neden Foundry'de değil, Python'da?** Proje başladığında (Temmuz 2026
başı) Foundry Local kataloğunda embedding modeli yoktu; retrieval bu yüzden
`sentence-transformers` ile Python tarafında kuruldu. 31 Temmuz 2026'da katalog
yeniden kontrol edildiğinde **artık var**: `qwen3-embedding-0.6b` (495 MB,
`task=embeddings`) ve `qwen3-embedding-8b`, SDK'da `model.get_embedding_client()`
ile erişilebiliyor. Yani mevcut kurulum artık bir zorunluluk değil, bir tercih —
ve bu tercih ölçülerek verildi. Foundry yolu bugün **kurulu ve çalışır** durumda:
`config.EMBED_BACKEND` tek satırda `"st"` ↔ `"foundry"` arasında geçiş yapıyor.

**Peki geçmeli miydik? Ölçtük: hayır.** `bench_embed.py`, aynı 31 parça ve 16 soru
üzerinde üç sistemi yalnızca retrieval açısından karşılaştırır — kararı LLM'in
gürültüsüne bırakmamak için üretim hiç çalıştırılmaz:

| | e5-small (384) | qwen3-emb ham (1024) | qwen3-emb instruct |
|---|---|---|---|
| **Kanıt ilk 3'te** (karar metriği) | **13/16 · %81** | 11/16 · %69 | **13/16 · %81** |
| Kanıt ilk 5'te | **14/16 · %88** | 12/16 · %75 | 13/16 · %81 |
| Doğru dosya ilk 3'te | **15/16 · %94** | 12/16 · %75 | 13/16 · %81 |
| Skor bandı (ilk 5 fark) | 0.035 | 0.129 | **0.137** |

Karar metriğinde (`TOP_K=3` olduğu için modelin gerçekten gördüğü bilgi) 20 kat
büyük model **berabere kaldı**, dosya bazında ise geride. İlginç olan, hatalarının
farklı yerlere düşmesi: Qwen kahvaltı saatlerini 5. sıradan 2.'ye ve açılış yılını
14. sıradan 3.'ye çekiyor (yani bilinen iki gerçek hatayı çözüyor), ama e5'in doğru
bulduğu "sahil barı" ve "havalimanı transferi" sorularını 12. ve 10. sıraya
düşürüyor. Skorları 4 kat daha geniş bir banda yaydığı da doğru — ama daha iyi
ayrıştırmak, daha doğru sıralamak anlamına gelmiyor.

**Ama bu kıyas taraflıydı — ikinci ölçüm.** Yukarıdaki test, parçaları
`CHUNK_MAX_CHARS=350` ile kurulmuş bir bilgi tabanı üzerinde yaptı; oysa o ayar
e5-small için optimize edilmişti. Qwen3-Embedding uzun bağlam için eğitilmiş bir
model, yani kıyas onun güçlü olduğu yeri hiç kullanmıyordu. `bench_chunking.py`
aynı iki modeli dört farklı parçalama ayarında ölçer (MRR = kanıtın sırasının
tersinin ortalaması; 1.0 = kanıt hep ilk sırada):

| Parçalama | Parça | Ort. uzunluk | e5-small MRR | qwen3-emb MRR |
|---|---|---|---|---|
| 350 / 120 / 80 *(mevcut, e5'e göre ayarlı)* | 31 | 212 | **0.729** | 0.728 |
| 700 / 200 / 120 | 20 | 322 | 0.699 | **0.746** |
| 1200 / 300 / 150 | 15 | 430 | 0.776 | **0.812** |
| belge bütünü (parçalama yok) | 9 | 718 | 0.840 | **0.854** |

Sonuç dürüstçe şu: **e5-small yalnızca kendi ayarında önde.** Diğer üç ayarda
Qwen daha isabetli sıralıyor. Yani "büyük model kazanamadı" demek eksik olurdu —
doğrusu, ilk kıyasın veri temsili bir modelin lehine hazırlanmıştı.

**Üçüncü ve belirleyici ölçüm: üretim ayarlarında, gerçekten bağlanmış haliyle.**
Önceki iki kıyas ya e5'e göre ayarlanmış bir parçalamada yapılmıştı ya da Foundry
yolu üretim koduna hiç bağlı değildi. Bu kez `rag_core.embed_texts()` Foundry
istemcisine gerçekten gitti, `ingest.py` 1024 boyutlu vektörlerle yeniden
çalıştırıldı ve ölçüm **dağıtılan ayarda** (9 parça) yapıldı:

| | e5-small (384) | qwen3-emb (1024) |
|---|---|---|
| **Kanıt ilk 3'te** (karar metriği) | **15/16 · %94** | 14/16 · %88 |
| Doğru dosya ilk 3'te | **14/16 · %88** | 13/16 · %81 |
| Kanıt ilk 1'de | 12/16 · %75 | **13/16 · %81** |
| MRR | 0.840 | **0.854** |
| 16 soruyu embedleme | **0.08 sn** | 11.75 sn |

Qwen ilk-1 ve MRR'de bir tık önde — ama bu üstünlük **ilk 3'ün içinde kalıyor**:
`TOP_K=3` ile modele giden bağlam değişmiyor, dolayısıyla cevaba yansımıyor.
Buna karşılık "Uçaktan indim, otele nasıl ulaşabilirim?" sorusunda doğru parçayı
2. sıradan 5.'ye düşürüyor; yani ilk 3'ün **dışına** çıkarıyor ve uçtan uca
`eval.py` turunda o soru ıskaya dönüşüyor (Foundry turu 14/17 · %82, aynı koşulda
e5 turu 15/17 · %88).

**Karar: `e5-small` kalıyor.** Karar metriğinde önde, sorgu başına ~0,73 sn daha
hızlı (cevap süresi ~2,5 sn iken bu ~%30'luk bir gecikme farkı demek), 20 kat
küçük ve GPU'daki sohbet modeliyle VRAM için yarışmıyor.

Planın 7. sayfası `qwen3-embedding-0.6b` öneriyor olsa da, bu proje için doğru
seçim ölçüme göre e5-small oldu; plana uymak için daha kötü bir sistem dağıtmanın
anlamı yok. Foundry yolu yine de **silinmedi**: çalışır, ölçülmüş ve tek satırla
geri açılabilir durumda. Belgeler uzayıp parça sayısı arttığında (Qwen'in uzun
bağlamda güçlendiğini `bench_chunking.py` zaten gösterdi) kıyas aynı
`bench_embed.py` ile tekrarlanabilir.

Arka uç değiştirildiğinde `python ingest.py` yeniden çalıştırılmalıdır: vektör
uzayı ve boyutu (384 ↔ 1024) değişir. Unutulursa `db.search()` bunu yakalar ve
sessizce saçma sonuç döndürmek yerine ne yapılacağını söyleyen bir hata verir.

**Asıl bulgu ise başka bir yerde:** her iki model de en iyi sonucu parçalama
*yokken* veriyor (MRR 0.73 → 0.84). Bu belgeler zaten kısa (ortalama 718
karakter); onları bölmek retrieval'a yardım etmiyor, zarar veriyor. Yani buradaki
en büyük kaldıraç embedding modeli değil, parçalama stratejisi — ve bu bulguya
göre hareket edildi (bir sonraki karar).

**Parçalama: bölmek zarar veriyordu.** Başlangıçta her paragraf kendi parçasına
düşüyordu (`CHUNK_MAX_CHARS=350` → 31 parça). Sezgi şuydu: küçük parça = keskin
embedding. Ölçüm bunu çürüttü — bir konunun cevabı komşu paragraftaki bağlamdan
koparıldığı için retrieval **zarar görüyordu**. Paragrafları 1000 karaktere kadar
*toparlayınca* bu projedeki belgeler (ortalama 718 karakter) tek parça kalıyor:

| | Eski (31 parça) | Yeni (9 parça) |
|---|---|---|
| Cevap doğruluğu (5 çalıştırma) | 13–14/17 · ort. **%80** | 14–16/17 · ort. **%88** |

1000 sınırı keyfi değil: kısa belgeleri bölmeyecek kadar büyük, uzun bir belgeyi
planın önerdiği "1–3 paragraflık" pasajlara bölecek kadar küçük. Yani aynı kod
sayfalarca belgede de doğru davranır — o zaman bir belge birden çok parça olur.

**`TOP_K = 3` — varsayım değil, ölçüm.** `tune_topk.py` taramasında isabet k=3'te
doyuyor; daha büyük k üretime yalnızca gürültü taşıyor.

---

## Değerlendirme

`eval.py`, elle hazırlanmış 17 soruluk bir set çalıştırır (doğrudan sorular,
dolaylı/parafraz sorular ve bilerek **bilgi tabanında olmayan** bir soru) ve iki
metriği **ayrı** ölçer:

| Metrik | Sonuç |
|---|---|
| **Cevap doğruluğu** | 14–16 / 17 · ortalama **%88** (5 çalıştırma) |
| **Retrieval isabeti** | 14 / 16 · **%88** (çalıştırmalar arası sabit) |

İki metriği ayırmak, bir hatanın nerede olduğunu söyler: cevap yanlış ama
retrieval doğruysa sorun **üretimde**, retrieval de ıskalamışsa sorun
**embedding/`TOP_K`** tarafındadır.

Cevap doğruluğunun bir aralık olarak verilmesinin sebebi `temperature=0.7`:
üretim örneklemeli olduğu için aynı soru çalıştırmalar arasında farklı
cevaplanabiliyor. Tek bir "en iyi" çalıştırmayı seçip raporlamak yerine gözlenen
aralık veriliyor. Retrieval ise deterministik olduğu için hiç oynamıyor.

Metriğin kendisi de bir kez sıkılaştırıldı. Önce bazı sorularda tek başına
`"var"` kabul anahtarıydı ve "en az bir anahtar geçsin" kuralı yüzünden model
yanlış bir şey söylese bile cümlesinde "var" geçtiği için **doğru sayılabiliyordu**.
Artık her soru kendine özgü terimi istiyor (jakuzi, hamam, aquapark, casino,
sahil barı…). Yukarıdaki tablo bu sıkı metrikle ölçülmüştür.

Değerlendirmede küçük ama önemli bir ayrıntı: anahtar kelimelerin bir kısmı
**bilerek kök** halinde yazılmıştır (`"bulunm"`, `"kabul edilm"`). Türkçe sondan
eklemeli olduğu için model "bulunmuyor / bulunmamaktadır / bulunmaz" diyebiliyor;
tam kelime araması bu doğru cevapları **yanlış** sayıyordu. Bu bir yazım hatası
değil, tam bir stemmer yerine kullanılan hafif bir kök eşleşmesidir (`eval.py`).

### Uç durumlar

Doğruluk ölçümünün yanına ayrı bir **dayanıklılık** bölümü eklendi: sistem
beklenmedik girdide ne yapıyor? `eval.py` beş uç durumu ayrı raporlar (5/5 geçiyor):

| Girdi | Beklenen davranış |
|---|---|
| `""` (boş) | Retrieval ve üretim **hiç çalışmamalı**; yönlendirme mesajı dönmeli |
| `"   "` (yalnız boşluk) | Aynı yol — `strip()` sonrası boş sayılmalı |
| `"Bana otelden bahset."` | Çok genel soruya bağlamdaki gerçeklere dayanan cevap |
| `"havuz"` | Tam cümle olmasa da doğru konuya gitmeli |
| `"Bugün hava nasıl olacak?"` | Bilgi tabanı dışı: uydurmamalı, "bilgim yok" demeli |

Bu sayı bilerek **ana yüzdelere karıştırılmadı**: ana set "doğru cevabı biliyor
mu?", bu bölüm "beklenmedik girdide güvenli mi?" sorusunu ölçüyor. Beşi de kolay
geçtiği için aynı tabloya konsaydı headline doğruluğu suni biçimde şişerdi.

Boş sorgunun kapıda durdurulması kozmetik bir kontrol değil: boş metin de
embed'lenebiliyor, `db.search()` yine de en benzer 3 parçayı döndürüyor (skorlar
anlamsız) ve model bu rastgele bağlamla **bir şeyler yazıyordu**. Artık
`rag.answer()` boş girdide retrieval'a hiç gitmiyor, kaynak listesi boş dönüyor —
hem doğru davranış hem de bedava (GPU'ya hiç gidilmiyor). Aynı sebeple
`assistant.py`'de boş Enter artık oturumu kapatmıyor; kazara basılan bir tuş
GPU'da yüklü modeli düşürmesin diye çıkış yalnızca `q`/`çıkış` ile.

### Güvenli başarısızlık

Kalan hatalar bu projede kasıtlı olarak **güvenli** tarafta bırakıldı: sistem
bilmediğinde uydurmuyor, "bilgim yok" diyor veya bağlama sadık kalıp eksik
cevaplıyor. Bunun bedeli birkaç puanlık doğruluk, karşılığı ise misafire yanlış
bilgi vermemek.

Bu davranış `rag.py` içindeki tek bir few-shot örneğiyle pekiştiriliyor ve o
örnek **bilerek olgu taşımıyor**. Sebebi bir deneyle öğrenildi: örneğe somut bir
sayı koyulduğunda ("200 oda"), model doğru parçayı getiremediği bir soruda o
sayıyı cevaba **sızdırdı** — yani few-shot örneğinin kendisi bir halüsinasyon
kaynağına dönüştü. Örnek yalnızca "bağlamda yoksa bilgi yok de" davranışını
öğreten, veri taşımayan bir güvenlik örneğine indirildi. Doğruluk aynı kaldı,
sızıntı ortadan kalktı.

---

## Bilinen sınırlamalar

- **Retrieval metriği fazla katı — %88 olduğundan kötümser.** Her soru için tek
  bir "doğru kaynak" bekliyor. Ama "otelde casino var mı" sorusunda beklenen
  `aktiviteler.txt` yerine `otel_genel.txt` geliyor; o belge de casino'dan
  bahsettiği için **cevap doğru çıkıyor**, metrik yine de ISKA sayıyor. Doğru
  çözüm, soru başına birden çok kabul edilebilir kaynak tanımlamak.
- **Benzerlik skorları birbirine çok yakın.** 31 parçalı eski yapılandırmada ilk
  beş parça arasındaki ortalama fark yalnızca 0.035'ti. Bunun tek suçlusunun
  embedding olmadığı ölçüldü: `qwen3-embedding-0.6b` skorları dört kat geniş bir
  banda yayıyor ama küçük parçalarda daha isabetli sıralamıyor. Parçaları
  büyütmek bu sorunu doğrudan hafifletti; kalan pay için bir yeniden-sıralama
  (reranker) adımı umut verici.
- **Bilgi tabanı küçük olduğu için "belge = parça" işe yarıyor.** Bu, sayfalarca
  belgeye ölçeklenmez: uzun bir belgenin tek embedding'i tüm konuların ortalaması
  olur ve hiçbir soruya güçlü eşleşmez. Parçalayıcı bunu zaten ele alıyor (1000
  karakteri aşan belgeler pasajlara bölünür) ama o rejim bu veriyle sınanmadı.
  Ayrıca `db.search()` tüm vektörleri belleğe alıp tek tek karşılaştırır — büyük
  N için gerçek bir vektör indeksi gerekir.
- **Model tavanı.** `qwen3-4b`, 6 GB VRAM'e sığan en iyi seçenek, ama günlük ve
  dolaylı ifadelerde ara sıra yanlış cümleye takılıyor. `llm.py` içindeki
  tekrar-kırpıcı döngüyü keser, içerik hatasını düzeltmez.
- **PDF:** yalnızca metin tabanlı. Taranmış (görüntü) PDF'ten metin çıkmaz; OCR yok.
- **Değerlendirme seti küçük** (17 soru) ve elle hazırlanmıştır; istatistiksel
  güven aralığı iddia etmez, yön gösterir.
- Tek kullanıcılı bir CLI'dir: web arayüzü, oturum hafızası ve çok-turlu diyalog yok.

## Sonraki adımlar

- **`eval.py`'da soru başına birden çok kabul edilebilir kaynak** tanımlamak —
  yukarıdaki casino örneğindeki yanlış ISKA'yı ortadan kaldırır.
- **Foundry embedding'i daha büyük bir bilgi tabanında yeniden ölçmek.** Bugünkü
  9 parçalık korpusta e5-small önde çıktı, ama `bench_chunking.py` Qwen'in parça
  sayısı arttıkça güçlendiğini gösteriyor. Arka uç zaten kurulu: tek satır
  değiştirip `bench_embed.py`'yi çalıştırmak yeterli.
- **Yeniden-sıralama (reranker)** ve basit bir web arayüzü (aynı `rag.answer()`
  üzerine) — ikisi de doğal sonraki adımlar.
