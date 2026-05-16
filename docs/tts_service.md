# TTS Service — Dokumentacija

Skripta `tts_service.py` skrbi za predvajanje vnaprej posnetih zvočnih opozoril za napravo SafeSteps.

## Struktura map

```
AIprojekt-SafeSteps/
    services/
        tts_service.py
    assets/
        audio/
            sl/
                zelena_luc_female.mp3
                zelena_luc_male.mp3
                rdeca_luc_female.mp3
                rdeca_luc_male.mp3
                ...
            en/
                green_light_female.mp3
                green_light_male.mp3
                ...
```

## Zahteve
 
- Python 3.8+
- `ffplay` (del ffmpeg paketa) — za predvajanje zvoka

Namestitev ffmpeg:
```bash
conda install -c conda-forge ffmpeg
```

## Način poimenovanja datotek

```
{kljuc}_{spol}.mp3
```

Primeri:
- `zelena_luc_female.mp3`
- `zelena_luc_male.mp3`
- `pozor_veja_female.mp3`

## Uporaba

### Osnovna uporaba

```python
from services.tts_service import predvajaj

# Ženski glas, slovensko (privzeto)
predvajaj("zelena_luc")

# Moški glas, slovensko
predvajaj("zelena_luc", spol="male")

# Ženski glas, angleško
predvajaj("green_light", spol="female", jezik="en")
```

### Parametri funkcije `predvajaj`

| Parameter | Tip | Privzeto | Opis |
|---|---|---|---|
| `kljuc` | `str` | — | Ime sporočila brez spola |
| `spol` | `str` | `"female"` | `"female"` ali `"male"` |
| `jezik` | `str` | `"sl"` | `"sl"` ali `"en"` |

## Dodajanje novih sporočil

1. Posnemite `.mp3` datoteko za oba spola (online tts, npr.: Narakeet)
2. Shranite v `assets/audio/sl/` ali `assets/audio/en/`
3. Dodajte ključ v množico `SOUNDS` v `tts_service.py`:

## Seznam sporočil

### Slovenščina (`sl`)

| Ključ | Besedilo |
|---|---|
| `zelena_luc` | "Zelena luč. Varno je za prečkanje." |
| `rdeca_luc` | "Rdeča luč. Počakajte." |
| `pozor_vozilo` | "Pozor, vozilo prihaja." |
| `pozor_veja` | "Pozor, veja." |
| `pozor_ovira_nad_glavo` | "Pozor, ovira nad glavo." |
| `pozor_stopnice_navzdol` | "Pozor, stopnice navzdol." |
| `pozor_stopnice_navzgor` | "Pozor, stopnice navzgor." |
| `naprava_vklopljena` | "Naprava je vklopljena." |
| `naprava_izklopljena` | "Naprava je izklopljena." |
| `baterija_prazna` | "Baterija je skoraj prazna." |

### Angleščina (`en`)

| Ključ | Besedilo |
|---|---|
| `green_light` | "Green light. Safe to cross." |
| `red_light` | "Red light. Please wait." |
| `warning_vehicle` | "Warning, vehicle approaching." |
| `warning_branch` | "Warning, branch." |
| `warning_overhead` | "Warning, overhead obstacle." |
| `warning_steps_down` | "Warning, steps going down." |
| `warning_steps_up` | "Warning, steps going up." |
| `device_on` | "Device is on." |
| `device_off` | "Device is off." |
| `battery_low` | "Battery is low." |
