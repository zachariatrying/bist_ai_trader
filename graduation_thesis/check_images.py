import hashlib
import os

images = [
    "borsaneuron_hisse_sorgu_real.png",
    "borsaneuron_scenario_ui.png",
    "senaryo_kume_profil.png",
    "prophet_forecast_real.png"
]

for img in images:
    path = os.path.join("images", img)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
            md5 = hashlib.md5(data).hexdigest()
            print(f"{img}: size={len(data)} bytes, md5={md5}")
    else:
        print(f"{img} does not exist!")
