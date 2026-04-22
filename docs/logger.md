# STM32 Data Logger

## Pregled

Projekt implementira binarni protokol za prenos podatkov iz senzorjev prek serijske povezave z STM32.

### Funkcionalnosti

**Implementirano:**

- Binarni prenos podatkov
- Download log datotek
- Parsing paketov
- Zaznavanje izgubljenih paketov

**To do:**

- Popravit snemanje surovega podatkovnega toka v `.bin`
- Vizualni dashboard — prikaz grafov

---

## Arhitektura

```
Serial (STM32)
      ↓
 Raw buffer
      ↓
 Frame extraction (sync 0xFFFF)
      ↓
 Byte unstuffing
      ↓
 CRC16 validacija
      ↓
 Packet parsing
      ↓
 ┌────────────┬──────────────┐
 │ print      │ save         │
 │ (live view)│ (.bin stream)│
 └────────────┴──────────────┘
```

---

## Struktura paketa

```
[Sync Marker: 0xFFFF]
[Packet Counter: uint8_t]
[Stuffed Payload]
```

> **Packet Counter:** vrednosti `0xFE` (254) in `0xFF` (255) so izpuščene, da ne pride do zmede s stuffing markerji.

**Payload vsebuje:**

| Polje                   | Tip        |
| ----------------------- | ---------- |
| Timestamp               | `uint32_t` |
| Velikost paketa         | `uint16_t` |
| Chunk podatki senzorjev | `[Chunk]*` |
| CRC16                   | `uint16_t` |

---

## Byte stuffing

_še treba dodati opis_

---

## Senzorji

### ID senzorjev

| ID     | Senzor        |
| ------ | ------------- |
| `0x01` | Žiroskop      |
| `0x02` | Pospeškometer |
| `0x03` | Magnetometer  |
| `0x05` | ToF           |

### Žiroskop, pospeškometer, magnetometer

- 3 vrednosti (X, Y, Z)
- Tip: `int16`
- 6 bytov na vzorec

### ToF

- Tip: `uint16`
- Enota: mm
- Neveljavna vrednost: `0xFFFF`

**Pri obdelavi:**

- `0xFFFF` pretvorimo v `NaN`
- Podatke združimo v signal
- Rekonstruiramo časovno os

---

## Prenos podatkov iz naprave

Trenutno sta podprta dva načina:

1. **Download all** — prenos vseh datotek
2. **Download by name** — prenos izbrane datoteke
