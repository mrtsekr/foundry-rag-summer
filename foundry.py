"""
Foundry Local'a tek giris noktasi.

NEDEN AYRI BIR MODUL: FoundryLocalManager bir TEKIL (singleton) — initialize()
ikinci kez cagrilirsa FoundryLocalException firlatir. Proje artik Foundry'yi
IKI yerden kullaniyor:

    llm.py      -> sohbet/uretim modeli  (qwen3-4b, GPU)
    rag_core.py -> embedding modeli      (qwen3-embedding-0.6b, CPU)

Ikisi de kendi initialize'ini cagirsaydi, ikisini birden kullanan her program
(eval.py, assistant.py, rag.py...) ikinci cagrida patlardi. Burada bir kez
kurulup paylasiliyor.

Execution provider (EP) kaydi da buraya tasindi: EP'ler PROCESS bazinda
kayitlidir ve kayit yapilmadan SDK yalnizca -generic-cpu varyantini gorur
(model sessizce CPU'da calisip 8-10x yavaslar). Ilk cagri EP ikililerini
indirebilir (ilk sefer ~13 dk); sonrasinda kayitli kalir.
"""

from foundry_local_sdk import Configuration, FoundryLocalManager

APP_NAME = "foundry_rag_summer"

_eps_hazir = False


def get_manager() -> FoundryLocalManager:
    """Paylasilan FoundryLocalManager'i dondurur; ilk cagrida kurar."""
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
    return FoundryLocalManager.instance


def ensure_eps(manager: FoundryLocalManager | None = None) -> FoundryLocalManager:
    """EP'leri bu process'e kaydeder (GPU varyantlari ancak bundan sonra gorunur).
    Process basina bir kez calisir."""
    global _eps_hazir
    manager = manager or get_manager()
    if not _eps_hazir:
        manager.download_and_register_eps()
        _eps_hazir = True
    return manager


def load_model(alias: str, gpu: bool = False):
    """Bir modeli katalogdan alir, (istenirse) GPU varyantini secer, gerekiyorsa
    indirir ve yukler. Yuklu IModel'i dondurur.

    gpu=False olan modeller (or. embedding) CPU'da kalir: sohbet modeli zaten
    6 GB VRAM'in ~4.5 GB'ini kullaniyor.
    """
    manager = ensure_eps() if gpu else get_manager()
    model = manager.catalog.get_model(alias)

    if gpu:
        varyant = _gpu_varyanti(model)
        if varyant is not None:
            model.select_variant(varyant)
        else:
            print(f"UYARI: {alias} icin GPU varyanti bulunamadi, CPU'da (yavas) calisabilir.")

    if not model.is_cached:
        print(f"{alias} indiriliyor (ilk sefer)...")
        model.download(lambda p: print(f"\r  {p:.0f}%", end="", flush=True))
        print()

    model.load()
    return model


def _gpu_varyanti(model):
    """CUDA GPU varyantini bul; yoksa herhangi bir GPU varyanti; o da yoksa None."""
    cuda = [v for v in model.variants if "cuda-gpu" in v.id.lower()]
    if cuda:
        return cuda[0]
    gpu = [v for v in model.variants
           if getattr(getattr(v.info, "runtime", None), "device_type", None) == "GPU"]
    return gpu[0] if gpu else None
