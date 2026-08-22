"""Tani: bir modelin tum varyantlarini ve hangisinin varsayilan secildigini gosterir.
Amac: qwen3-4b neden CPU'ya dustu, GPU (cuda) varyanti hangisi -> anlamak."""
from foundry_local_sdk import Configuration, FoundryLocalManager

config = Configuration(app_name="foundry_rag_summer")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

for alias in ["qwen3-4b", "qwen3.5-4b"]:
    print(f"\n{'='*60}\n{alias}\n{'='*60}")
    model = manager.catalog.get_model(alias)
    secili = model.id  # varsayilan secili varyantin id'si
    for v in model.variants:
        info = v.info
        ep = getattr(info, "execution_provider", "?")
        cached = getattr(info, "cached", "?")
        isaret = "  <== VARSAYILAN SECILI" if v.id == secili else ""
        print(f"  id={v.id}")
        print(f"      EP={ep}  cached={cached}{isaret}")
