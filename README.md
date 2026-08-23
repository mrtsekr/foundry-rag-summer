# Deniz Yıldızı Resort - Offline Türkçe RAG Asistanı

Tamamen yerel çalışan Türkçe bir soru-cevap (RAG) asistanı. Bir otelin
belgelerini okur ve misafir sorularını yalnızca o belgelere dayanarak yanıtlar.
Bulut yok, API anahtarı yok, veri cihazdan çıkmaz.

Bir aylık yaz okulu projesi. Amacı RAG'in her parçasını (parçalama, embedding,
retrieval, üretim) hazır bir çerçeveye sarılmadan elle kurmak ve her ayarı
ölçerek seçmek.

**Tanıtım sayfası:** <https://mrtsekr.github.io/foundry-rag-summer/>
Aynı içeriğin görsel hâli: mimari şeması, ölçüm grafikleri ve 21 gerçek
çalıştırmada arama. Tek HTML dosyası, sunucu istemez. `python site/sunucu.py`
ile açarsan sayfa canlı moda geçer ve istediğin soruyu gerçekten modele sorarsın.

| | |
|---|---|
| Üretim (chat) | [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/) + `qwen3-4b`, yerel GPU'da (RTX 3060 Laptop, 6 GB) |
| Retrieval | `sentence-transformers` + `intfloat/multilingual-e5-small` (384 boyut) |
| Depo | Python'un yerleşik `sqlite3`'ü, ayrı vektör veritabanı yok |
| Bilgi tabanı | 9 belge → 9 parça (`.txt` · `.md` · `.pdf`) |

---

## Ne yapıldı

RAG hattının tamamı elle yazıldı. LangChain, LlamaIndex ya da benzeri bir
çerçeve, ve Chroma/FAISS gibi bir vektör veritabanı kullanılmadı. Bunun sebebi
tercih değil amaç: her adımın ne yaptığını görmek.

Elle yazılan kısımlar:

- **Parçalayıcı** (`ingest.py` → `chunk_text`): paragraf öncelikli, toparlayıcı
  (greedy packing). Paragrafları 1000 karaktere kadar birleştirir, sınırı aşan
  belgeleri örtüşmeli pasajlara böler.
- **Retrieval** (`db.py` → `search`, `rag_core.py` → `cosine_similarity`):
  sorgu vektörü ile bütün parçalar arasında kosinüs benzerliği, en iyi `TOP_K`.
- **Depolama** (`db.py`): SQLite şeması ve embedding'lerin `float32` BLOB olarak
  saklanması. Ayrı bir vektör sunucusu yok.
- **İstem ve few-shot** (`rag.py`): bağlam yerleştirme, kaynak gösterimi, ve
  bağlamda cevap yoksa "bilgim yok" davranışını öğreten tek bir güvenlik örneği.
- **Tekrar kırpıcı** (`llm.py`): küçük Qwen modellerinin girdiği tekrar
  döngüsünü çıktıda keser.
- **Değerlendirme takımı** (`eval.py`): 17 soru, cevap doğruluğu ve retrieval
  isabeti ayrı ayrı, artı 5 uç durum testi.
- **Kıyas araçları** (`bench/`): embedding arka uçları, parçalama ayarları ve
  `TOP_K` taraması.
- **Tanıtım sayfası** (`site/`): tek dosya, düz HTML/CSS/JS, build adımı yok.

Hazır alınanlar: Foundry Local (model çalıştırma), `sentence-transformers`
(embedding modeli), `sqlite3` (stdlib), `pypdf`, `numpy`, `torch`.

Bu depodaki her ayar bir ölçümle seçildi. Aşağıdaki tablolar o ölçümlerin
kendisi; reddedilen denemeler de yazılı.

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

## Masaüstü ve telefon

Komut satırı dışında iki kullanım biçimi var. İkisi de aynı `rag.answer()`
üzerinde çalışır, ayrı bir kod yolu yoktur.

**Masaüstü.** Depo kökündeki `Deniz Yildizi Asistani.bat` dosyasına çift tıkla.
Sunucuyu başlatır ve sayfayı Edge'in uygulama kipinde açar: adres çubuğu ve
sekme yok, kendi penceresi ve görev çubuğu girişi var. Edge yoksa Chrome, o da
yoksa varsayılan tarayıcı denenir.

Bu dosya **depo klasörünün içinde** durmalıdır; tek başına indirilip
çalıştırılamaz, yanındaki `site/sunucu.py`'yi arar. Açılmadan önce üç ön koşulu
denetler ve eksik olanı hemen söyler — beklemeye bırakmaz:

| Eksik olan | Ne yapmalı |
|-----------|-----------|
| `site\sunucu.py` yok | Depoyu klonla, `.bat`'ı depo klasöründen çalıştır |
| Python yok | Python 3.12 kur, `## Kurulum` adımlarını uygula |
| `rag_store.db` yok | `python ingest.py` çalıştır |

Model yüklenmesini beklemez: port açılır açılmaz pencereyi açar, bekleme
sayfanın kendi açılış ekranında geçer. Sunucu 20 saniyede dinlemeye başlamazsa
en olası sebep Foundry Local servisidir (`foundry service status`).

Arayüz tanıtım sayfasının kendisidir ama `/uygulama` yolundan açılır ve o kipte
tanıtım bölümleri gizlenir; geriye soru kutusu, cevap ve kaynaklar kalır. Ayrı
bir HTML tutulmadı, çünkü iki sürüm zamanla birbirinden kayar.

**Telefon.** Model telefonda çalışamaz; `qwen3-4b` bu bilgisayarın GPU'sunda.
Telefon desteği, telefonun aynı ağ üzerinden bu bilgisayara bağlanması demektir:

```bash
python site/sunucu.py --ag
```

Sunucu açılışta telefonun yazacağı adresi gösterir
(`http://<bu-makinenin-ip>:8000/uygulama`). Telefon tarayıcısında açıp "Ana
ekrana ekle" dersen simgesiyle ve tam ekran açılır.

`--ag` varsayılan değildir ve olmamalıdır: `0.0.0.0`'a bağlanmak asistanı o
ağdaki herkese açar. İnternet yine devrede değil, belgeler makineden çıkmıyor;
yalnızca soru ve cevap yerel ağda dolaşıyor.

**Mobil yerleşimi cihazsız doğrulamak.** `http://127.0.0.1:8000/telefon`
uygulamayı sabit telefon ölçülerinde bir çerçeve içinde açar
(`site/telefon.html`); 393×852, 412×915 ve 375×667 arasında geçilebilir.
İçerideki sayfa bir kopya değil, iframe doğrudan `/uygulama`'yı yükler: aynı
sunucu, aynı model, çalışan bir asistan. Çerçeve tamamen CSS, görsel dosyası
yok. Duyarlı yerleşimin dar ekranda bozulmadığı elde cihaz olmadan böyle
denetlenebiliyor.

**Sunucunun verdiği adresler.** `python site/sunucu.py` çalışırken:

| Adres | Ne gösterir |
|-------|-------------|
| `/` | Tanıtım sayfasının tamamı — ölçümler, kararlar, canlı deneme |
| `/uygulama` | Yalnız asistan: tanıtım bölümleri gizli, soru kutusu ve cevap kalır |
| `/telefon` | `/uygulama`'yı sabit telefon ölçülerinde çerçeveler; mobil yerleşimi cihazsız denetlemek için |
| `/saglik` | JSON: sunucu ayakta mı, model yüklendi mi, hangi modeller |
| `/sor` | POST, JSON `{"soru": "..."}` — cevap, kaynaklar ve süre döner |

Sayfa `/saglik`'i yoklar; sunucu ayaktaysa **canlı** kipe geçip gerçekten soru
sorar, değilse kayıtlı 21 çalıştırmayı gösterir. Aynı dosya iki kipte de
çalıştığı için GitHub Pages'te de açılır, orada kayıtlı kipte kalır.

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
| `site/` | Tanıtım sayfası: `index.html`, `kayitlar.json`, `kayit_uret.py`, `sunucu.py`, `telefon.html` |
| `Deniz Yildizi Asistani.bat` | Masaüstü başlatıcı: ön koşulları denetler, sunucuyu açar, uygulamayı Edge'in uygulama kipinde başlatır |

`tani/` altındakiler tek seferlik teşhis scriptleri ama duruyorlar, çünkü
belgelenen VRAM ve süre sayılarını bunlar üretti.

---

## Teknik kararlar

### Chat modeli: `qwen3-4b`

Kararı donanım verdi. Aday `qwen3.5-4b` (5.36 GB) 6 GB'lık karta sığmadı.

| Model | Yükleme | VRAM tepe | Ortalama cevap |
|---|---|---|---|
| `qwen3-4b` | 11.9 sn | 4516 / 6144 MiB | **2.7 sn** |
| `qwen3.5-4b` | 29.6 sn | 5903 / 6144 MiB | 115.1 sn |

`qwen3.5-4b` paylaşımlı belleğe taştı ve düşünme bloğunun içinde takılıp geçerli
cevap üretemedi. Ölçüm `tani/compare_chat.py`.

### GPU'yu elle seçmek gerekiyor

Foundry SDK, çalışma sağlayıcıları (EP) bu sürece kayıtlı değilken sessizce
`-generic-cpu` varyantını seçiyor. Model doğru çalışır ama sürünür ve hiçbir hata
mesajı vermez; sorun yalnızca yavaşlık olarak görünür. `llm.py` önce EP'leri
kaydedip sonra `cuda-gpu` varyantını açıkça seçiyor: **21.6 sn → 2.5 sn**, aynı
soru, aynı model, aynı bilgisayar. Teşhis `tani/diag_variants.py` ve
`tani/diag_eps.py`.

### Embedding çok dilli olmalı

İngilizce merkezli `all-MiniLM-L6-v2` Türkçe sorularda (havuz, evcil hayvan)
yanlış parça getirdi ve model uydurdu. `paraphrase-multilingual-MiniLM-L12-v2`
bunu düzeltti; retrieval için özel eğitilmiş `multilingual-e5-small` dolaylı
sorularda daha da iyi oldu (retrieval isabeti %88 → %94). Üçü de 384 boyut
olduğu için şema hiç değişmedi. e5 modelleri ön ek ister: sorguya `query: `,
parçaya `passage: ` (`rag_core.py`).

### `temperature = 0.7`, sezgiye ters ama doğru

Model aynı cümleyi 20 kez tekrarlıyordu. Suçlu sanılan sıcaklık aslında çözümdü:
düşük sıcaklıkta (0.3) üretim aynı token dizisine saplanıyordu. 0.7 döngüyü
kırdı, cevap bağlama dayalı olduğu için kalite bozulmadı.

### `frequency_penalty` kullanılmadı

Tekrar sorununa bariz çözüm gibi görünüyordu. 0.3 ve 0.6 ile denendi, ikisinde de
işi kötüleştirdi: cevapları boşalttı ve bir soruda "kumarhane yok"u "var"a
çevirerek tehlikeli bir halüsinasyon üretti. Doğru araç sıcaklıktı.

### Parçaları bölmemek

Başlangıçta her paragraf ayrı parçaya düşüyordu (`CHUNK_MAX_CHARS=350`, 31 parça)
ve sezgi "küçük parça = keskin embedding" diyordu. Ölçüm bunu çürüttü: bir
konunun cevabı komşu paragraftaki bağlamdan koparılıyordu.

| | Eski (31 parça) | Yeni (9 parça) |
|---|---|---|
| Cevap doğruluğu (5 çalıştırma) | 13-14 / 17, ort. %80 | 14-16 / 17, ort. **%88** |

1000 sınırı keyfi değil: bu korpustaki belgeleri (ortalama 718 karakter) bölmeyecek
kadar büyük, uzun bir belgeyi birkaç paragraflık pasajlara bölecek kadar küçük.
Aynı kod sayfalarca belgede de doğru davranır, o zaman bir belge birden çok parça
olur.

### Foundry'nin embedding modeline geçilmedi

Proje başlarken Foundry Local tarafında kullanılabilecek bir embedding modeli
bulunamadı ve retrieval Python tarafında kuruldu. Bunun bir sebebi de arama
biçimiydi: SDK'da modeller takma adla çekiliyor (`catalog.get_model(alias)`), yani
takma adını bilmediğin bir model senin için görünmez oluyor. Temmuz sonunda
SDK üzerinden yeniden bakıldığında `qwen3-embedding-0.6b` (495 MB) ve
`qwen3-embedding-8b` erişilebilir çıktı.

Yani mevcut kurulum artık zorunluluk değil, tercih. Tercih üç ölçümle verildi.

**Birinci ölçüm** (31 parça, e5 için optimize edilmiş ayar) e5'i açık ara önde
gösterdi, ama bu kıyas taraflıydı: Qwen3-Embedding uzun bağlam için eğitilmiş bir
model ve o ayar onun güçlü olduğu yeri hiç kullanmıyordu.

**İkinci ölçüm** aynı iki modeli dört farklı parçalama ayarında karşılaştırdı
(`bench/bench_chunking.py`). MRR = kanıtın sırasının tersinin ortalaması; 1.0
kanıtın hep ilk sırada olması demek.

| Parçalama | Parça | Ort. uzunluk | e5 hit@3 | qwen hit@3 | e5 MRR | qwen MRR |
|---|---|---|---|---|---|---|
| 350 / 120 / 80 | 25 | 264 | 11/16 | **13/16** | 0.665 | **0.795** |
| 700 / 200 / 120 | 12 | 538 | 14/16 | 14/16 | 0.747 | **0.792** |
| 1200 / 300 / 150 | 9 | 718 | **15/16** | 14/16 | 0.840 | **0.854** |
| parçalama yok | 9 | 718 | **15/16** | 14/16 | 0.840 | **0.854** |

Qwen dört ayarın dördünde de daha iyi MRR veriyor, yani doğru parçayı listenin
tepesine taşımakta iyi. Ama `TOP_K=3` ile çalışan bir sistemde belirleyici olan
MRR değil, doğru parçanın ilk 3'e girip girmediği.

**Üçüncü ve belirleyici ölçüm** üretim ayarında yapıldı: Foundry yolu gerçekten
`rag_core.embed_texts()` üzerinden bağlandı, `ingest.py` 1024 boyutlu vektörlerle
yeniden çalıştırıldı, ölçüm dağıtılan ayarda (9 parça) alındı.

| Ölçüt (16 soru, 9 parça) | e5-small (384) | qwen3-emb (1024) |
|---|---|---|
| Kanıt ilk 3'te (karar metriği) | **15/16 · %94** | 14/16 · %88 |
| Doğru dosya ilk 3'te | **14/16 · %88** | 13/16 · %81 |
| Kanıt ilk 1'de | 12/16 · %75 | **13/16 · %81** |
| MRR | 0.840 | **0.854** |
| 16 soruyu embedleme | **0.08 sn** | 11.75 sn |

20 kat büyük model karar metriğinde geride kaldı ve sorgu başına ~0.73 saniye
ekledi. Somut örnek: havalimanı transferi sorusunda doğru parçayı 2. sıradan
5.'ye düşürdü, yani ilk 3'ün dışına. Foundry yolu silinmedi,
`config.EMBED_BACKEND` tek satırda geri açıyor ve bilgi tabanı büyüdüğünde aynı
kıyas tekrarlanabiliyor.

### `TOP_K = 3`

`bench/tune_topk.py` taramasında isabet k=3'te doyuyor, daha büyük k üretime
yalnızca gürültü taşıyor.

---

## Değerlendirme

`eval.py` elle hazırlanmış 17 soruluk bir set çalıştırır: doğrudan sorular,
dolaylı/parafraz sorular ve bilerek bilgi tabanında olmayan bir soru. İki metrik
ayrı ölçülür.

| Metrik | Sonuç |
|---|---|
| Cevap doğruluğu | 13-16 / 17, ortalama **%84** (4 çalıştırma) |
| Retrieval isabeti | 15 / 16, **%94** (çalıştırmalar arası sabit) |
| Uç durumlar | 5 / 5 (ayrı ölçülür) |

İki metriği ayırmak bir hatanın nerede olduğunu söyler: cevap yanlış ama
retrieval doğruysa sorun üretimde, retrieval de ıskalamışsa sorun
embedding veya `TOP_K` tarafında.

Cevap doğruluğu tek sayı değil aralık, çünkü üretim örneklemeli
(`temperature = 0.7`). En iyi tur seçilmedi, gözlenen aralık yazıldı. Retrieval
deterministik olduğu için hiç oynamıyor.

`eval.py` süre ölçmüyor, o yüzden tabloda yok. Cevap süresi ayrı ölçüldü ve tek
bir sayı değil: `site/kayitlar.json` içindeki 21 gerçek çalıştırmanın ortalaması
1.33 sn, medyanı 1.14, aralığı 0.85-3.42 sn. GPU düzeltmesi bölümündeki 2.5 sn
başka bir ölçümden, o karşılaştırmanın kendi öncesi/sonrası çiftinden geliyor.

### Metrik iki kez sıkılaştırıldı

Her ikisinde de raporlanan sayı düştü. Ölçüm ne kadar gevşekse sonuç o kadar
anlamsız.

Birinci turda tek başına `"var"` kabul anahtarı olan sorular temizlendi. "En az
bir anahtar geçsin" kuralı yüzünden model yanlış bir şey söylese bile cümlesinde
"var" geçtiği için doğru sayılabiliyordu. Artık her soru kendine özgü terimi
istiyor: jakuzi, hamam, aquapark, casino, sahil barı.

İkinci tur bir kod incelemesinden geldi, temizlik eksik kalmıştı: "Akşam et yemek
istiyorum, uygun bir restoran var mı?" sorusunda `"var"` hâlâ kabul anahtarıydı
ve soru zaten "var mı?" ile bittiği için neredeyse her cevap geçiyordu.

### Uç durumlar

Doğruluk ölçümünün yanında ayrı bir dayanıklılık bölümü var: sistem beklenmedik
girdide ne yapıyor? `eval.py` beş uç durumu ayrı raporlar, beşi de geçiyor.

| Girdi | Beklenen davranış |
|---|---|
| `""` (boş) | Retrieval ve üretim hiç çalışmamalı, yönlendirme mesajı dönmeli |
| `"   "` (yalnız boşluk) | Aynı yol, `strip()` sonrası boş sayılmalı |
| `"Bana otelden bahset."` | Çok genel soruya bağlamdaki gerçeklere dayanan cevap |
| `"havuz"` | Tam cümle olmasa da doğru konuya gitmeli |
| `"Bugün hava nasıl olacak?"` | Bilgi tabanı dışı, uydurmamalı |

Bu beş test ana yüzdelere karıştırılmadı. Ana set "doğru cevabı biliyor mu", bu
bölüm "beklenmedik girdide güvenli mi" sorusunu ölçüyor; beşi de kolay geçtiği
için aynı tabloya konsaydı doğruluk suni biçimde şişerdi.

Boş sorgunun kapıda durdurulması kozmetik bir kontrol değil. Boş metin de
embed'lenebiliyor, `db.search()` yine en benzer 3 parçayı döndürüyor (skorlar
anlamsız) ve model bu rastgele bağlamla bir şeyler yazıyordu. Artık
`rag.answer()` boş girdide retrieval'a hiç gitmiyor. Aynı sebeple
`assistant.py`'de boş Enter oturumu kapatmıyor; kazara basılan bir tuş GPU'da
yüklü modeli düşürmesin diye çıkış yalnızca `q` ile.

### Kapanmamış düşünme bloğu

Qwen3 düşünme modunda çalışıyor ve `/no_think` her zaman tutmuyor. `_run()`
başlangıçta yalnızca `</think>` varsa bloğu ayıklıyordu. Model bloğu açıp
`max_tokens` sınırına takılarak kapatamadığında blok olduğu gibi kullanıcıya
gidiyordu: "Otelde aquapark var mı?" sorusuna cevap olarak modelin İngilizce iç
sesi döndü.

Hata site için gerçek çalıştırma kayıtları üretilirken çıktı ama üretim kodunu
etkiliyordu, CLI asistanında da olabilirdi. Artık blok açılmış ama kapanmamışsa
cevap boş döndürülüyor, `generate()` bunu kötü cevap sayıp bir kez yeniden
deniyor.

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

- **Retrieval'da kalan tek ıska gerçek bir ıska.** Metrik başta her soru için tek
  bir doğru kaynak bekliyordu ve bazı bilgiler birden çok belgede geçtiği için
  asistan doğru cevap verdiği hâlde ıska sayılıyordu. Bu düzeltildi: iki soru
  artık birden çok kabul edilebilir kaynak tanımlıyor (casino sorusu
  `aktiviteler.txt` ya da `otel_genel.txt`, balayı sorusu `balayi.pdf` ya da
  `politikalar.txt`). Ölçülen fark 14/16'dan 15/16'ya. Geriye kalan tek ıska
  metrik sorunu değil: "Denize karşı bir şeyler içebileceğim sahil barı var mı?"
  sorusunda `yeme_icme.txt` yerine `balayi.pdf`, `havuz_plaj.txt` ve
  `ulasim.txt` geliyor. Dolaylı sorularda embedding'in gerçekten şaşırdığı yer
  burası.
- **Benzerlik skorları birbirine çok yakın.** 21 çalıştırmanın 63 skorunun hepsi
  0.77-0.87 aralığına sıkışıyor, yani yüksek skor tek başına "doğru parça geldi"
  demek değil. 31 parçalı eski yapılandırmada ilk beş parça arasındaki ortalama
  fark yalnızca 0.035'ti. Parçaları büyütmek bunu hafifletti; kalan pay için bir
  yeniden sıralama adımı gerekir.
- **"Belge = parça" yaklaşımı bu korpusa özel.** Sayfalarca belgeye ölçeklenmez:
  uzun bir belgenin tek embedding'i tüm konuların ortalaması olur ve hiçbir
  soruya güçlü eşleşmez. Parçalayıcı bunu ele alıyor (1000 karakteri aşan
  belgeler pasajlara bölünüyor) ama o rejim bu veriyle sınanmadı. Ayrıca
  `db.search()` tüm vektörleri belleğe alıp tek tek karşılaştırıyor; büyük N için
  gerçek bir vektör indeksi gerekir.
- **Model tavanı.** `qwen3-4b` 6 GB'a sığan en iyi seçenek ama günlük ve dolaylı
  ifadelerde ara sıra yanlış cümleye takılıyor. `llm.py` içindeki tekrar kırpıcı
  döngüyü keser, içerik hatasını düzeltmez.
- **PDF yalnızca metin tabanlı.** Taranmış (görüntü) PDF'ten metin çıkmaz, OCR yok.
- **Değerlendirme seti küçük** (17 soru) ve elle hazırlandı. İstatistiksel güven
  aralığı iddia etmiyor, yön gösteriyor.
- **Oturum hafızası yok.** Her soru bağımsız, çok turlu diyalog kurulmadı.
- **İptal yalnızca istemci tarafında.** Uygulamadaki "iptal" düğmesi ve 60
  saniyelik zaman aşımı isteği bırakır, arayüzü kurtarır; ama sunucudaki üretimi
  durdurmaz. Model o soruyu bitirene kadar `_kilit`'i tuttuğu için bir sonraki
  soru gecikebilir. Gerçek iptal, üretimi parça parça kesebilen bir akış
  arayüzü gerektirir.

## Sonraki adımlar

- **Embedding kıyasını daha büyük bir bilgi tabanında tekrarlamak.** Bugünkü 9
  parçalık korpusta e5-small önde çıktı, ama yukarıdaki parçalama tablosu Qwen'in
  parça sayısı arttıkça güçlendiğini gösteriyor. Eksik olan kod değil veri: arka
  uç kurulu, `config.EMBED_BACKEND` tek satır. Yeni belgeler gerçek olduğunda
  anlamlı olur; sırf ölçüm yapmak için sentetik belge üretmek hem sonucu hem
  anlatıyı bozar.
- **Yeniden sıralama (reranker).** Bu ölçekte işe yaramaz: toplam 9 parça var ve
  `TOP_K = 3`. Reranker'ın işi çok sayıda aday arasından iyi olanı yukarı çekmek,
  9 adayın olduğu yerde sıralanacak bir şey yok. Üstelik ikinci bir model
  (cross-encoder) cevap süresine eklenir. Parça sayısı birkaç yüze çıkarsa
  gündeme gelir.
- **Çok turlu diyalog.** Şu an her soru bağımsız. Önceki soruya atıf yapan
  ("peki orada kahvaltı kaçta?") sorular için bir soru yeniden yazma adımı
  gerekir. Değerlendirme seti de buna göre genişletilmeli, yoksa iyileşme
  ölçülemez.

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
- Wang ve ark., [*Multilingual E5 Text Embeddings*](https://arxiv.org/abs/2402.05672) (2024) — `query:` / `passage:` ön ek kuralı buradan
- [Foundry Local SDK dokümantasyonu](https://learn.microsoft.com/azure/ai-foundry/foundry-local/reference/reference-sdk) — EP kaydı ve varyant seçimi

Tanıtım sayfasındaki bileşenler [21st.dev](https://21st.dev) kaynaklarından
uyarlandı (Timeline, Hero Highlight, AI Agent Pipeline, animated-beam,
bento-grid). Hepsi React bileşeniydi, bu projede düz HTML/CSS'e taşındı.

## Lisans

Kod [MIT lisansı](LICENSE) altındadır.

Bu lisans `docs/` klasörünü kapsamaz. Oradaki belgeler üçüncü bir tarafın herkese
açık içeriğinden uyarlanmış ve anonimleştirilmiş örnek verilerdir; yalnızca
gösterim amacıyla bulunuyorlar. Ayrıntısı [`docs/NOTICE.md`](docs/NOTICE.md)
dosyasında, yani verinin yanında duruyor. (`ingest.py` bu dosyayı bilgi tabanına
almaz; `docs/` altındaki NOTICE, README, LICENSE ve CHANGELOG adlı dosyalar
içerik değil açıklama sayılır.)
