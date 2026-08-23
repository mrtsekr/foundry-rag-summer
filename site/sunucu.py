"""Tanitim sayfasi icin kucuk yerel sunucu — "canli mod".

Neden var: site/index.html'e gomulu 21 kayit sabit bir havuz. Kendi
bilgisayarinda ISTEDIGIN soruyu sorabilmen icin gercek RAG hattinin
calismasi gerekiyor. Bu dosya rag.answer()'i ince bir HTTP katmaniyla
sariyor, baska hicbir sey yapmiyor.

Neden stdlib: requirements.txt'e tek bir paket eklemesin diye. Python'in
yerlesik http.server'i bu is icin fazlasiyla yeterli — burasi tek kisilik
bir gelistirme sunucusu, internete acilmasi dusunulmedi.

    python site/sunucu.py           -> http://127.0.0.1:8000  (yalniz bu makine)
    python site/sunucu.py --ag      -> ayni Wi-Fi'daki telefondan da erisilir

Sayfa ayni koken uzerinden servis edildigi icin CORS'a hic girilmiyor.
Sunucu kapaliyken index.html gomulu kayit havuzuna geri duser; yani bu
dosya olmadan da sayfa calisir.

--ag hakkinda: model telefonda calisamaz (qwen3-4b bu makinenin GPU'sunda).
Telefon destegi, telefonun ayni ag uzerinden BU makineye baglanmasi demek.
Varsayilan 127.0.0.1'dir cunku 0.0.0.0'a baglanmak asistani agdaki HERKESE
acar. Bilincli bir tercih olsun diye bayrakla istenir. Internet yine yok:
belgeler makineden cikmiyor, yalnizca soru ve cevap yerel agda dolasiyor.
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
import db  # noqa: E402
import rag  # noqa: E402
from llm import load_chat  # noqa: E402

AG_KIPI = "--ag" in sys.argv
ADRES = "0.0.0.0" if AG_KIPI else "127.0.0.1"
PORT = 8000
INDEX = SITE_DIR / "index.html"

# Ana ekrana eklenince uygulama gibi acilsin diye (telefonda tarayici
# cubuklari gizlenir). Yalnizca bu sunucu servis eder; GitHub Pages'e
# kopyalanan surumde referans verilmez.
MANIFEST = {
    "name": "Deniz Yildizi Asistani",
    "short_name": "Deniz Yildizi",
    "start_url": "/uygulama",
    "scope": "/",
    "display": "standalone",
    "background_color": "#E9EDEE",
    "theme_color": "#0B6971",
    "icons": [{
        "src": ("data:image/svg+xml,"
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
                "%3Crect width='64' height='64' rx='14' fill='%230B6971'/%3E"
                "%3Cpath d='M12 40c6-6 10 6 16 0s10 6 16 0' stroke='%23E9EDEE'"
                " stroke-width='4' fill='none' stroke-linecap='round'/%3E"
                "%3Ccircle cx='32' cy='22' r='5' fill='%23E9EDEE'/%3E%3C/svg%3E"),
        "sizes": "any", "type": "image/svg+xml", "purpose": "any",
    }],
}

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

        # /uygulama ayni dosyayi verir; sayfa yolu gorup uygulama kipine gecer.
        # Ayri bir HTML tutmuyoruz ki iki surum zamanla birbirinden kaymasin.
        if yol in ("/", "/index.html", "/uygulama", "/uygulama/"):
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

        if yol == "/manifest.webmanifest":
            _json(self, 200, MANIFEST)
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
    # Cikti tamponda beklemesin: kullanici model yuklenirken ne oldugunu
    # ve --ag ile telefonun yazacagi adresi ANINDA gormeli. Boruya
    # yazarken print() blok tamponlu oldugu icin bu gerekli.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

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

    # Embedding modeli sentence-transformers tarafinda TEMBEL yukleniyor: ilk
    # arama onu da indirip kurdugu icin ilk canli soru 11,7 sn suruyordu
    # (sonrakiler 2,5 sn). Bir kez bosa arama yapip o bedeli baslangica
    # tasiyoruz; boylece sayfadan gelen ILK soru da normal hizda cevaplaniyor.
    print("Embedding isindiriliyor...")
    try:
        conn = db.connect()
        db.search(conn, "isinma", 1)
        conn.close()
    except Exception as e:
        print("  (isindirma atlandi: %s)" % e)

    sunucu = ThreadingHTTPServer((ADRES, PORT), Islem)
    print()
    print("  Canli mod hazir:  http://127.0.0.1:%d" % PORT)
    if AG_KIPI:
        # Telefonun yazacagi adresi bulup gosterelim; kullanici ipconfig'e
        # bakmak zorunda kalmasin.
        import socket
        try:
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(("8.8.8.8", 80))       # paket gitmez, yalnizca yerel IP'yi ogrenir
            yerel_ip = _s.getsockname()[0]
            _s.close()
        except OSError:
            yerel_ip = "<bu-makinenin-ip-adresi>"
        print()
        print("  TELEFONDAN:       http://%s:%d/uygulama" % (yerel_ip, PORT))
        print("  Ayni Wi-Fi'da olmalisiniz. Telefon tarayicisinda acip")
        print("  'Ana ekrana ekle' derseniz uygulama gibi calisir.")
        print()
        print("  DIKKAT: --ag ile asistan bu agdaki HERKESE acik.")
        print("  Guvenmediginiz bir agda kullanmayin.")
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
