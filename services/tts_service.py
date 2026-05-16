import subprocess
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT_DIR, "assets", "audio")

SOUNDS = {
    "sl": {
        "zelena_luc",
        "rdeca_luc",
        "pozor_vozilo",
        "pozor_veja",
        "pozor_ovira_nad_glavo",
        "pozor_stopnice_navzdol",
        "pozor_stopnice_navzgor",
        "naprava_vklopljena",
        "naprava_izklopljena",
        "baterija_prazna",
    },
    "en": {
        "green_light",
        "red_light",
        "warning_vehicle",
        "warning_branch",
        "warning_overhead",
        "warning_steps_down",
        "warning_steps_up",
        "device_on",
        "device_off",
        "battery_low",
    },
}

SPOLI = {"female", "male"}


def predvajaj(kljuc: str, spol: str = "female", jezik: str = "sl") -> None:
    """
    Predvaja zvočno sporočilo.

    Args:
        kljuc: Ime sporočila brez spola (npr. 'zelena_luc').
        spol:  'female' ali 'male' (privzeto 'female').
        jezik: 'sl' ali 'en' (privzeto 'sl').
    """
    if jezik not in SOUNDS:
        print(f"Napaka: neznan jezik '{jezik}'.")
        return

    if spol not in SPOLI:
        print(f"Napaka: neznan spol '{spol}'. Uporabi 'female' ali 'male'.")
        return

    if kljuc not in SOUNDS[jezik]:
        print(f"Napaka: neznan ključ '{kljuc}' za jezik '{jezik}'.")
        return

    ime_datoteke = f"{kljuc}_{spol}.mp3"
    pot = os.path.join(AUDIO_DIR, jezik, ime_datoteke)

    if not os.path.exists(pot):
        print(f"Napaka: datoteka ne obstaja: {pot}")
        return

    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", pot],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

""" Primer uporabe. Kasneje odstrani."""
if __name__ == "__main__":
    predvajaj("zelena_luc", spol="female", jezik="sl")
    predvajaj("rdeca_luc")