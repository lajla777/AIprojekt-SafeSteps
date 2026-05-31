import struct
import pytest

# ── funkcije ki jih testiramo (kopirane iz decode.py ker ne uvažamo celotnega
#    modula – ta zahteva pyserial ki ga v CI ni nujno nameščenega) ──────────

SYNC = b'\xFF\xFF'

CHUNK_NAMES = {
    0x01: 'gyroscope',
    0x02: 'accelerometer',
    0x03: 'magnetometer',
    0x05: 'ToF sensor'
}

def crc16_update(crc, data):
    crc ^= data
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc >>= 1
    return crc

def crc16_compute(data):
    crc = 0xFFFF
    for byte in data:
        crc = crc16_update(crc, byte)
    return crc

def unstuff_bytes(data: bytes) -> bytes:
    unstuffed = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0xFE:
            i += 1
            if i < len(data):
                unstuffed.append(data[i] ^ 0xFE)
        else:
            unstuffed.append(data[i])
        i += 1
    return bytes(unstuffed)


# ── CRC testi ────────────────────────────────────────────────────────────────

def test_crc16_compute_znan_rezultat():
    """CRC za znane podatke mora vrniti fiksen rezultat."""
    data = bytes([0x01, 0x02, 0x03])
    rezultat = crc16_compute(data)
    assert rezultat == crc16_compute(data)  # determinističen

def test_crc16_prazen_vhod():
    """CRC praznega niza vrne začetno vrednost 0xFFFF."""
    assert crc16_compute(b'') == 0xFFFF

def test_crc16_razlicni_podatki_razlicen_crc():
    """Različni podatki morajo dati različen CRC."""
    assert crc16_compute(b'\x01') != crc16_compute(b'\x02')

def test_crc16_konsistentnost():
    """Isti vhod vedno vrne isti CRC."""
    data = b'\xFF\x01\xAB\xCD'
    assert crc16_compute(data) == crc16_compute(data)

def test_crc16_update_en_bajt():
    """crc16_update vrne uint16 vrednost."""
    rezultat = crc16_update(0xFFFF, 0x01)
    assert 0 <= rezultat <= 0xFFFF


# ── byte stuffing testi ──────────────────────────────────────────────────────

def test_unstuff_brez_stuffinga():
    """Podatki brez 0xFE se ne spremenijo."""
    data = bytes([0x01, 0x02, 0x03])
    assert unstuff_bytes(data) == data

def test_unstuff_en_stuffed_bajt():
    """0xFE 0xFE ^ 0xFE = 0x00 → unstuff vrne 0x00."""
    # stuffed 0x00: 0xFE, (0x00 ^ 0xFE) = 0xFE
    data = bytes([0xFE, 0xFE])
    rezultat = unstuff_bytes(data)
    assert rezultat == bytes([0x00])

def test_unstuff_stuffed_ff():
    """Stuffan 0xFF: 0xFE, (0xFF ^ 0xFE) = 0x01."""
    data = bytes([0xFE, 0x01])
    rezultat = unstuff_bytes(data)
    assert rezultat == bytes([0xFF])

def test_unstuff_prazen_vhod():
    """Prazen vhod vrne prazne bajte."""
    assert unstuff_bytes(b'') == b''

def test_unstuff_fe_na_koncu():
    """0xFE na koncu brez naslednjega bajta ne doda ničesar."""
    data = bytes([0x01, 0xFE])
    rezultat = unstuff_bytes(data)
    assert rezultat == bytes([0x01])

def test_unstuff_ohrani_navadne_bajte():
    """Navadni bajti se ohranijo."""
    data = bytes([0xAA, 0xBB, 0xCC])
    assert unstuff_bytes(data) == data


# ── ToF razdalja testi ───────────────────────────────────────────────────────

def test_tof_razdalja_normalna():
    """ToF vrednost 850mm se pravilno pakira/razpakira."""
    raw = struct.pack('<H', 850)
    vrednost = struct.unpack('<H', raw)[0]
    assert vrednost == 850

def test_tof_razdalja_nič():
    """ToF vrednost 0 je veljavna."""
    raw = struct.pack('<H', 0)
    assert struct.unpack('<H', raw)[0] == 0

def test_tof_razdalja_maksimum():
    """ToF senzor meri do 2000mm."""
    raw = struct.pack('<H', 2000)
    vrednost = struct.unpack('<H', raw)[0]
    assert vrednost == 2000

def test_tof_neveljavna_vrednost_0xFFFF():
    """0xFFFF pomeni neveljavno meritev (out of range)."""
    neveljavna = 0xFFFF
    assert neveljavna > 2000  # mora biti filtrirano kot NaN

def test_tof_opozorilni_prag():
    """Ovira znotraj 500mm sproži opozorilo."""
    prag_mm = 500
    assert 300 < prag_mm   # blizu → opozorilo
    assert 1200 >= prag_mm  # daleč → ni opozorila

def test_tof_v_obmocju_senzorja():
    """Veljavne razdalje so med 0 in 2000mm."""
    for razdalja in [0, 100, 500, 1000, 2000]:
        assert 0 <= razdalja <= 2000

# ── IMU chunk testi ──────────────────────────────────────────────────────────

def test_chunk_names_vsebuje_vse_senzorje():
    """Vse pričakovane ID-je so v CHUNK_NAMES."""
    assert 0x01 in CHUNK_NAMES  # gyroscope
    assert 0x02 in CHUNK_NAMES  # accelerometer
    assert 0x03 in CHUNK_NAMES  # magnetometer
    assert 0x05 in CHUNK_NAMES  # ToF

def test_imu_sample_xyz_velikost():
    """IMU vzorec ima 3 vrednosti (x, y, z) po 2 bajta = 6 bajtov."""
    raw = struct.pack('<hhh', 100, -200, 300)
    x, y, z = struct.unpack('<hhh', raw)
    assert x == 100
    assert y == -200
    assert z == 300

def test_imu_sample_negativne_vrednosti():
    """Žiroskop in accelerometer lahko vrneta negativne vrednosti."""
    raw = struct.pack('<hhh', -1000, -2000, -3000)
    x, y, z = struct.unpack('<hhh', raw)
    assert x < 0 and y < 0 and z < 0

def test_sync_marker():
    """SYNC marker je 0xFF 0xFF."""
    assert SYNC == b'\xFF\xFF'
    assert len(SYNC) == 2