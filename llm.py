"""
Hafta 3 · Uretim (generation) katmani — Foundry Local + qwen3-4b (GPU)
RAG'in 'G' adimi: bulunan baglami alip qwen3-4b ile Turkce cevap uretir.

Bu modul, model karsilastirmasinda ogrendigimiz TUM tuzaklari icerir:
  - manager.download_and_register_eps(): EP'ler kayitli olmazsa SDK sadece CPU
    varyantini sunar ve model CPU'da (yavas) calisir. Ilk cagri uzun surer.
  - select_variant(cuda-gpu): varsayilan -generic-cpu geliyor; GPU'yu acikca seceriz.
  - /no_think + max_tokens: Qwen3 dusunme modunu kapatir ve tekrar dongusunu sinirlar.
  - frequency_penalty KULLANMIYORUZ: qwen3-4b'de cevap kalitesini bozdugunu gorduk.

load_chat() modeli bir kez yukler (ilk cagri ~10-15 sn + ilk sefer EP/indirme).
"""

import re

import config
import foundry

# Uretim ayarlari (RAG icin)
MAX_TOKENS = 300      # RAG cevaplari kisa; olasi runaway'i erken keser
# Tekrar dongusu ASLINDA dusuk sicaklik (greedy) belirtisiydi: 0.3'te model ayni
# token dizisine saplaniyordu. 0.7 dongulari kirar; cevap baglama dayali oldugu
# icin kaliteyi bozmaz. NOT: frequency_penalty KULLANMIYORUZ -> qwen3-4b'de
# cevabi bozdugunu, hatta 'yok'u 'var'a cevirdigini gorduk (0.3 ve 0.6'da).
TEMPERATURE = 0.7
TOP_P = 0.9

_chat = None  # yuklenen sohbet istemcisi (tekil)
_model = None


def load_chat():
    """qwen3-4b'yi GPU'da yukler ve ayarlanmis sohbet istemcisini dondurur.
    Ikinci cagrida ayni istemciyi verir (yeniden yuklemez).

    Foundry baglantisi/EP kaydi/GPU varyant secimi foundry.py'ye tasindi:
    embedding arka ucu da Foundry olabildigi icin manager tekili paylasilmali.
    """
    global _chat, _model
    if _chat is not None:
        return _chat

    model = foundry.load_model(config.CHAT_MODEL, gpu=True)

    chat = model.get_chat_client()
    chat.settings.max_tokens = MAX_TOKENS
    chat.settings.temperature = TEMPERATURE
    chat.settings.top_p = TOP_P

    _model, _chat = model, chat
    return chat


def _trim_repetition(text: str) -> str:
    """Kucuk modeller bazen ayni cumleyi tekrarlayarak donguye girer. Cumlelere
    boler; daha once gorulmus bir cumle tekrar gelirse orada keser (dongu tail'ini
    atar). Boylece kullaniciya 5 kez tekrarlanan cumle gitmez."""
    cumleler = re.split(r"(?<=[.!?])\s+", text.strip())
    gorulen = set()
    sonuc = []
    for c in cumleler:
        anahtar = c.strip().lower()
        if not anahtar:
            continue
        if anahtar in gorulen:      # tekrar -> dongu basladi, kes
            break
        gorulen.add(anahtar)
        sonuc.append(c.strip())
    return " ".join(sonuc)


def _run(chat, messages) -> str:
    """Verilen mesajlarla tek bir uretim yapar; <think> blogunu ayiklar ve
    olasi tekrar dongusunu kirpar."""
    parcalar = []
    for chunk in chat.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parcalar.append(delta)
    cevap = "".join(parcalar)
    if "</think>" in cevap:
        cevap = cevap.split("</think>", 1)[1]
    return _trim_repetition(cevap)


def _kotu_cevap(cevap: str, user: str) -> bool:
    """Cevap bos mu ya da girdiyi (soruyu) AYNEN tekrarliyor mu (echo)? Bu iki
    durumda cevabi 'kotu' sayip bir kez yeniden deneriz."""
    c = cevap.strip().lower()
    if not c:
        return True
    # Model bazen cevap uretmeyip girdinin bir parcasini aynen geri yazar (echo).
    if len(c) > 12 and c in user.lower():
        return True
    return False


def generate(system: str, user: str, examples=None, _retry: bool = True) -> str:
    """Tek turlu uretim: system + (opsiyonel few-shot ornekleri) + user mesajiyla
    qwen3-4b'den Turkce cevap alir.

    examples: [(ornek_soru, ornek_cevap), ...] -> modele "boyle cevapla" ornegi
              gosterir (few-shot). Cevap bicimini ve baglama sadik kalmayi ogretir.
    _retry:   cevap bos ya da echo ise bir kez yeniden dener.
    Dusunme modunu kapatmak icin system'e '/no_think' eklenir.
    """
    chat = load_chat()
    messages = [{"role": "system", "content": f"{system} /no_think"}]
    for ornek_soru, ornek_cevap in (examples or []):
        messages.append({"role": "user", "content": ornek_soru})
        messages.append({"role": "assistant", "content": ornek_cevap})
    messages.append({"role": "user", "content": user})

    cevap = _run(chat, messages)
    if _retry and _kotu_cevap(cevap, user):
        cevap2 = _run(chat, messages)          # bir kez daha dene
        if not _kotu_cevap(cevap2, user):
            return cevap2
        return cevap or cevap2                 # ikisi de kotuyse bos olmayani ver
    return cevap
