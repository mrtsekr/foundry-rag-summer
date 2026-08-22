"""Tani-2: EP'leri bu process'e kaydettikten SONRA GPU varyantlari cikiyor mu?"""
from foundry_local_sdk import Configuration, FoundryLocalManager

config = Configuration(app_name="foundry_rag_summer")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

print("=== discover_eps() ===")
for ep in manager.discover_eps():
    print(f"  {ep}")

print("\n=== download_and_register_eps() (indiriliyse hizli/no-op) ===")
res = manager.download_and_register_eps()
print(f"  sonuc: {res}")

print("\n=== EP kaydindan SONRA qwen3-4b varyantlari ===")
model = manager.catalog.get_model("qwen3-4b")
secili = model.id
for v in model.variants:
    rt = getattr(v.info, "runtime", None)
    ep = getattr(rt, "execution_provider", "?") if rt else "?"
    dev = getattr(rt, "device_type", "?") if rt else "?"
    mark = "  <== SECILI" if v.id == secili else ""
    print(f"  id={v.id}  device={dev}  EP={ep}  cached={v.info.cached}{mark}")
