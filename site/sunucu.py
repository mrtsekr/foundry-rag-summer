"""Tanitim sayfasi icin kucuk yerel sunucu — "canli mod".

Neden var: site/index.html'e gomulu 21 kayit sabit bir havuz. Kendi
bilgisayarinda ISTEDIGIN soruyu sorabilmen icin gercek RAG hattinin
calismasi gerekiyor. Bu dosya rag.answer()'i ince bir HTTP katmaniyla
sariyor, baska hicbir sey yapmiyor.

Neden stdlib: requirements.txt'e tek bir paket eklemesin diye. Python'in
yerlesik http.server'i bu is icin fazlasiyla yeterli — burasi tek kisilik
bir gelistirme sunucusu, internete acilmasi dusunulmedi.

    python site/sunucu.py
    -> http://127.0.0.1:8000

Sayfa ayni koken uzerinden servis edildigi icin CORS'a hic girilmiyor.
Sunucu kapaliyken index.html gomulu kayit havuzuna geri duser; yani bu
dosya olmadan da sayfa calisir.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SITE_DIR.parent))  # proje kokundeki modulleri gor

import config  # noqa: E402
import rag  # noqa: E402
from llm import load_chat  # noqa: E402

ADRES = "127.0.0.1"
PORT = 8000
INDEX = SITE_DIR / "index.html"

# Modele ayni anda tek istek gitsin. Hem Foundry istemcisinin is parcacigi
# guvenligine bel baglamamak icin, hem de 6 GB'lik kartta paralel uretim
# denemenin zaten anlami olmadigi icin.
_kilit = threading.Lock()


def _json(handler: BaseHTTPRequestHandler, kod: int, govde: dict) -> None:
    ham = json.dumps(govde, ensure_ascii=False).encode("utf-8")
    handler.send_response(kod)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(ham)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(ham)


class Islem(BaseHTTPRequestHandler):
    server_version = "DenizYildiziSunucu/1.0"

    def log_message(self, bicim, *args):  # varsayilan gurultuyu kis
        sys.stderr.write("  %s\n" % (bicim % args))

    # ---------------- GET ----------------
    def do_GET(self):
        yol = self.path.split("?", 1)[0]

        if yol in ("/", "/index.html"):
            try:
                ham = INDEX.read_bytes()
            except OSError:
                self.send_error(404, "index.html bulunamadi")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(ham)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(ham)
            return

        if yol == "/saglik":
            # Sayfa acilista bunu yokluyor. Donen JSON'daki "canli" alani
            # gorulurse arayuz canli moda geciyor.
            _json(self, 200, {
                "canli": True,
                "chat_modeli": config.CHAT_MODEL,
                "embedding": config.EMBED_MODEL,
                "top_k": config.TOP_K,
            })
            return

        self.send_error(404, "yok")

    # ---------------- POST ----------------
    def do_POST(self):
        if self.path.split("?", 1)[0] != "/sor":
            self.send_error(404, "yok")
            return

        try:
            uzunluk = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            uzunluk = 0
        if uzunluk <= 0 or uzunluk > 8192:
            _json(self, 400, {"hata": "Gecersiz istek govdesi."})
            return

        try:
            istek = json.loads(self.rfile.read(uzunluk).decode("utf-8"))
            soru = (istek.get("soru") or "").strip()
        except (ValueError, UnicodeDecodeError):
            _json(self, 400, {"hata": "JSON cozulemedi."})
            return

        # Bos sorguyu rag.answer zaten kapida yakaliyor; burada da erken donup
        # GPU'ya hic gitmiyoruz.
        if not soru:
            _json(self, 200, {
                "canli": True, "bos": True,
                "cevap": rag.BOS_SORU_MESAJI, "kaynaklar": [], "sure": 0.0,
            })
            return

        with _kilit:
            basla = time.perf_counter()
            try:
                cevap, hits = rag.answer(soru)
            except Exception as e:  # modeli/DB'yi sessizce yutma, durumu bildir
                _json(self, 500, {"hata": "Cevap uretilemedi: %s" % e})
                return
            sure = time.perf_counter() - basla

        # hits: (skor, id, metin, kaynak_dosya)
        kaynaklar = [{"dosya": kaynak, "skor": round(float(skor), 3)}
                     for skor, _id, _metin, kaynak in hits]

        _json(self, 200, {
            "canli": True,
            "soru": soru,
            "cevap": cevap,
            "kaynaklar": kaynaklar,
            "sure": round(sure, 2),
        })


def main() -> int:
    if not INDEX.exists():
        print("index.html bulunamadi: %s" % INDEX)
        return 1

    print("Model yukleniyor (ilk sefer ~10-15 sn)...")
    try:
        load_chat()
    except Exception as e:
        print("Model yuklenemedi: %s" % e)
        print("Foundry Local kurulu ve calisir durumda mi?")
        return 1

    sunucu = ThreadingHTTPServer((ADRES, PORT), Islem)
    print()
    print("  Canli mod hazir:  http://%s:%d" % (ADRES, PORT))
    print("  Chat: %s | Embedding: %s | TOP_K: %d"
          % (config.CHAT_MODEL, config.EMBED_MODEL, config.TOP_K))
    print("  Durdurmak icin Ctrl+C")
    print()
    try:
        sunucu.serve_forever()
    except KeyboardInterrupt:
        print("\nKapatiliyor.")
    finally:
        sunucu.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
