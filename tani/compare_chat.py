"""
Chat modeli karsilastirmasi (GPU, duzeltilmis): qwen3-4b  vs  qwen3.5-4b
Amac: Turkce kalitesi + hiz + VRAM'i ADIL kosulda kiyaslayip karar vermek.

Onceki denemede ogrenilen ve burada duzeltilen 3 sorun:
  1) EP kaydi: SDK, EP'ler process'e kayitli degilse sadece CPU varyantini sunar.
     -> manager.download_and_register_eps() cagiriyoruz (ilk sefer uzun, sonra hizli).
  2) Varyant secimi: varsayilan -generic-cpu geliyor. -> cuda-gpu varyantini
     select_variant() ile ACIKCA seciyoruz (GPU'da calissin).
  3) Uretim kontrolu: Qwen3 dusunme modu + sinirsiz uretim -> tekrara giriyordu.
     -> system prompt'a "/no_think", ayrica max_tokens ve temperature ayari.

Calistir:  python compare_chat.py
Ilk calistirmada ~8 GB GPU model iner.
"""

import subprocess
import time

from foundry_local_sdk import Configuration, FoundryLocalManager

MODELLER = ["qwen3-4b", "qwen3.5-4b"]

# Uretim ayarlari (ikisine de ayni uygulanir -> adil kiyas)
MAX_TOKENS = 400
TEMPERATURE = 0.3
TOP_P = 0.9
FREQ_PENALTY = 0.6      # tekrar/dongu onleme (kucuk Qwen3 modelleri tekrara giriyor)
PRESENCE_PENALTY = 0.3  # ayni konuya saplanip kalmayi azaltir
NO_THINK = "/no_think"  # Qwen3 dusunme modunu kapatan yumusak anahtar

SORULAR = [
    {
        "ad": "RAG - bilgi cekme",
        "system": "Sen bir otel asistanisin. SADECE sana verilen bilgiye dayanarak kisa ve Turkce cevap ver. Bilgi yoksa 'bilmiyorum' de.",
        "user": "Bilgi: Kahvalti her sabah 07:00-10:30 arasinda restoranda servis edilir. Havuz 09:00-19:00 acik.\n\nSoru: Kahvaltiyi en gec kacta yiyebilirim?",
    },
    {
        "ad": "Turkce akicilik",
        "system": "Kisa ve dogal bir Turkce ile cevap ver.",
        "user": "Deniz kenarindaki bu oteli bir turiste iki cumleyle, samimi bir dille tanit.",
    },
    {
        "ad": "Talimat takibi + nezaket",
        "system": "Kisa ve dogal bir Turkce ile cevap ver.",
        "user": "Bir misafir gec check-out istiyor ama otel dolu. Nazik bir dille, Turkce, tam 2 cumlelik bir yanit yaz.",
    },
]


def gpu_vram_used_mib():
    """nvidia-smi ile o an kullanilan GPU bellegini (MiB) okur."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def pick_gpu_variant(model):
    """Modelin CUDA GPU varyantini bulur; yoksa herhangi bir GPU varyanti."""
    cuda = [v for v in model.variants if "cuda-gpu" in v.id.lower()]
    if cuda:
        return cuda[0]
    gpu = [v for v in model.variants
           if getattr(getattr(v.info, "runtime", None), "device_type", None) == "GPU"]
    return gpu[0] if gpu else None


def indirme_progress(alias):
    """Log'u sisirmemek icin ilerlemeyi sadece her %10'da bir yazan callback."""
    son = [-10]
    def cb(p):
        if p >= son[0] + 10 or p >= 100:
            son[0] = int(p)
            print(f"    {alias} indiriliyor: {p:.0f}%", flush=True)
    return cb


def sor(chat, system, user):
    """Tek soru sorar; /no_think ekli system + user ile. (cevap, sure_sn) doner."""
    messages = [
        {"role": "system", "content": f"{system} {NO_THINK}"},
        {"role": "user", "content": user},
    ]
    t0 = time.perf_counter()
    parcalar = []
    for chunk in chat.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parcalar.append(delta)
    return "".join(parcalar), time.perf_counter() - t0


def temizle(metin):
    """<think> blogunu cevaptan ayikla. Kapanmadiysa (dusunme icinde takildiysa)
    o kismin tamamini at ve isaretle."""
    if "</think>" in metin:
        return metin.split("</think>", 1)[1].strip()
    if "<think>" in metin:
        # Acildi ama kapanmadi -> model dusunme icinde takildi/tekrara girdi
        return "[!! dusunme icinde takildi, gecerli cevap yok]"
    return metin.strip()


def modeli_dene(manager, alias):
    print(f"\n{'='*70}\nMODEL: {alias}\n{'='*70}", flush=True)
    sonuc = {"alias": alias, "cevaplar": []}

    model = manager.catalog.get_model(alias)

    # --- GPU varyantini acikca sec ---
    gpu_v = pick_gpu_variant(model)
    if gpu_v is None:
        print("  !! GPU varyanti bulunamadi (EP kaydi eksik olabilir), atlaniyor.")
        return sonuc
    model.select_variant(gpu_v)
    rt = gpu_v.info.runtime
    print(f"  Secilen varyant: {gpu_v.id}  (device={rt.device_type}, EP={rt.execution_provider})", flush=True)

    if not model.is_cached:
        print("  Indiriliyor (ilk sefer)...", flush=True)
        model.download(indirme_progress(alias))

    vram_bos = gpu_vram_used_mib()
    print("  Yukleniyor...", flush=True)
    t0 = time.perf_counter()
    model.load()
    sonuc["yukleme_sn"] = time.perf_counter() - t0
    vram_yuk = gpu_vram_used_mib()
    sonuc["vram_yukleme_mib"] = vram_yuk
    print(f"  Yukleme: {sonuc['yukleme_sn']:.1f} sn | VRAM yukleme sonrasi: {vram_yuk} MiB "
          f"(bos: {vram_bos} MiB, agirlik ~{(vram_yuk - vram_bos) if (vram_yuk and vram_bos) else '?'} MiB)", flush=True)

    chat = model.get_chat_client()
    chat.settings.max_tokens = MAX_TOKENS
    chat.settings.temperature = TEMPERATURE
    chat.settings.top_p = TOP_P
    chat.settings.frequency_penalty = FREQ_PENALTY
    chat.settings.presence_penalty = PRESENCE_PENALTY

    vram_peak = vram_yuk or 0
    for s in SORULAR:
        cevap, sure = sor(chat, s["system"], s["user"])
        cevap = temizle(cevap)
        v = gpu_vram_used_mib()
        if v:
            vram_peak = max(vram_peak, v)
        sonuc["cevaplar"].append({"ad": s["ad"], "cevap": cevap, "sure_sn": sure})
        print(f"\n  --- [{s['ad']}]  ({sure:.1f} sn) ---", flush=True)
        print("  " + cevap.replace("\n", "\n  "), flush=True)

    sonuc["vram_peak_mib"] = vram_peak
    model.unload()
    print(f"\n  {alias} bosaltildi. VRAM peak (KV dahil): {vram_peak} MiB / 6144 MiB", flush=True)
    return sonuc


def main():
    config = Configuration(app_name="foundry_rag_summer")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("EP'ler kaydediliyor (ikililer inmisse hizli)...", flush=True)
    res = manager.download_and_register_eps()
    print(f"  EP kayit: success={res.success}, registered={res.registered_eps}", flush=True)

    tum = [modeli_dene(manager, alias) for alias in MODELLER]

    print(f"\n\n{'#'*70}\n# OZET\n{'#'*70}")
    print(f"{'Model':<14}{'Yukleme':>10}{'VRAM yuk':>10}{'VRAM peak':>11}{'Ort.cevap':>11}")
    for r in tum:
        c = r["cevaplar"]
        ort = sum(x["sure_sn"] for x in c) / len(c) if c else 0
        print(f"{r['alias']:<14}{r.get('yukleme_sn',0):>9.1f}s"
              f"{r.get('vram_yukleme_mib','?'):>10}{r.get('vram_peak_mib','?'):>11}{ort:>10.1f}s")
    print("\n> VRAM peak 6144'e cok yakinsa/asiyorsa o model VRAM'de sikisik demektir.")
    print("> Turkce kalitesini yukaridaki 3 cevaptan sen degerlendir; hiz sayilari yukarida.")


if __name__ == "__main__":
    main()
