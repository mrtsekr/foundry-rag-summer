"""
Hafta 1 - "Hello Model" testi
Amac: Foundry Local runtime'inin gercekten calistigini dogrulamak.
Henuz RAG YOK - sadece yerel bir LLM yukleyip tek bir soru soruyoruz.

Calistirmak icin (proje klasorunde, venv aktifken):
    python hello_model.py
"""

from foundry_local_sdk import Configuration, FoundryLocalManager

# 1) SDK'yi baslat. app_name sadece bir etiket; model onbellegi bu isimle iliskilenir.
config = Configuration(app_name="foundry_rag_summer")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance  # tekil (singleton) yonetici nesnesi

# 2) Katalogdan bir sohbet modeli sec. qwen2.5-0.5b = en kucuk/hizli chat modeli (~0.5GB).
#    GPU'n oldugu icin Foundry uygun GPU varyantini otomatik secebilir.
model = manager.catalog.get_model("qwen2.5-0.5b")

# 3) Model diskte yoksa indir (ilk seferde). Sonraki calistirmalarda bu adim atlanir.
if not model.is_cached:
    print("Model indiriliyor (ilk sefere ozel)...")
    model.download(lambda p: print(f"\r  Indirme: {p:.1f}%", end="", flush=True))
    print()

# 4) Modeli belege (GPU/CPU) yukle. Bu, cikarim icin hazir hale getirir.
print("Model yukleniyor...")
model.load()

# 5) Sohbet istemcisi al ve tek bir soru sor.
chat = model.get_chat_client()
messages = [
    {"role": "system", "content": "Sen yardimci bir asistansin. Kisa ve net cevap ver."},
    {"role": "user", "content": "Merhaba! Tek cumleyle kendini tanit."},
]

print("\nCevap: ", end="", flush=True)
# Streaming: cevap token token gelir; geldikce ekrana yazariz.
for chunk in chat.complete_streaming_chat(messages):
    # Son parca bos "choices" ile gelebilir (bitis/istatistik parcasi) -> once kontrol et.
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print("\n")

# 6) Temizlik: modeli bellekten bosalt (agirliklar diskte kalir, tekrar indirilmez).
model.unload()
print("Bitti - Foundry Local calisiyor! ✅")
