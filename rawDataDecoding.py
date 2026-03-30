import struct
import numpy as np

def crc16_update(crc, data):
    crc ^= data
    for i in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc = crc >> 1
    return crc

def crc16_compute(data):
    crc = 0xFFFF
    for byte in data:
        crc = crc16_update(crc, byte)
    return crc

def unstuff_bytes(data):
    unstuffed = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0xFE:
            i += 1
            if i >= len(data):
                break
            unstuffed.append(0xFE ^ data[i])
        else:
            unstuffed.append(data[i])
        i += 1
    return bytes(unstuffed)

def debug_crc(data, index):
    raw = data[2:]
    received_crc = struct.unpack('<H', raw[-2:])[0]
    received_crc_be = struct.unpack('>H', raw[-2:])[0]

    payload = unstuff_bytes(raw[:-2])
    full_unstuffed = unstuff_bytes(raw)

    candidates = {
        "payload (unstuffed, brez CRC)":        crc16_compute(payload),
        "payload + sync (unstuffed, brez CRC)": crc16_compute(b'\xFF\xFF' + payload),
        "raw (brez CRC, brez unstuff)":         crc16_compute(raw[:-2]),
        "raw + sync (brez unstuff)":            crc16_compute(data[:2] + raw[:-2]),
        "full unstuffed (z CRC)":               crc16_compute(full_unstuffed),
    }

    print(f"\nPaket {index}:")
    print(f"  received CRC (LE): 0x{received_crc:04X}")
    print(f"  received CRC (BE): 0x{received_crc_be:04X}")
    for name, val in candidates.items():
        match = " ✓ MATCH" if val == received_crc or val == received_crc_be else ""
        print(f"  {name}: 0x{val:04X}{match}")

def parse_packet(data):
    if data[0:2] != b'\xFF\xFF':
        return None

    if len(data) < 5:
        return None

    packet_counter = data[2]  # ← packet counter, not stuffed

    raw = data[3:]  # ← payload začne pri bajtu 3, ne 2
    if len(raw) < 4:
        return None

    # CRC je stuffan skupaj s payloadom - unstuffaj vse skupaj
    payload = unstuff_bytes(raw)

    if len(payload) < 8:
        return None

    # CRC je zadnja 2 bajta unstuffanega payloada
    received_crc = struct.unpack('<H', payload[-2:])[0]
    computed_crc = crc16_compute(payload[:-2])

    if received_crc != computed_crc:
        return None

    timestamp = struct.unpack('<I', payload[0:4])[0]
    packet_size_enc = struct.unpack('<H', payload[4:6])[0]

    chunks_data = payload[6:-2]
    pos = 0
    chunks = {}

    while pos < len(chunks_data):
        if pos + 4 > len(chunks_data):
            break
        chunk_id = chunks_data[pos]
        chunk_size_enc = struct.unpack('<H', chunks_data[pos+1:pos+3])[0]
        chunk_size = chunk_size_enc + 1
        reserved = chunks_data[pos+3]
        chunk_data = chunks_data[pos+4:pos+4+chunk_size]

        samples = []
        for i in range(0, chunk_size, 6):
            if i+6 <= len(chunk_data):
                x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
                samples.append((x, y, z))

        chunks[chunk_id] = samples
        pos += 4 + chunk_size

    return {'timestamp': timestamp, 'packet_counter': packet_counter, 'chunks': chunks}

with open("dataTestBIN.bin", "rb") as f:
    data = f.read()

print(f"velikost datoteke: {len(data)} bajtov")

sync = b'\xFF\xFF'
positions = []
pos = 0
while pos < len(data) - 1:
    if data[pos:pos+2] == sync:
        positions.append(pos)
    pos += 1

print(f"najdenih paketov: {len(positions)}")

# --- Parsanje ---
packets = []
for i in range(len(positions)):
    start = positions[i]
    end = positions[i+1] if i+1 < len(positions) else len(data)
    packet = parse_packet(data[start:end])
    if packet is not None:
        packets.append(packet)

print(f"uspešno parsiranih paketov: {len(packets)}")

gyro_samples = []
accel_samples = []
mag_sampples = []
timestamps = []

for packet in packets:
    timestamps.append(packet['timestamp'])
    chunks = packet['chunks']
    if 1 in chunks:
        gyro_samples.extend(chunks[1])
    if 2 in chunks:
        accel_samples.extend(chunks[2])
    else:
        accel_samples.append((0, 0, 0))
    if 3 in chunks:
        mag_sampples.extend(chunks[3])

gyro_array = np.array([np.array(s) for s in gyro_samples], dtype=np.int16)
accel_array = np.array([np.array(s) for s in accel_samples], dtype=np.int16)
mag_array = np.array([np.array(s) for s in mag_sampples], dtype=np.int16)
timestamps_array = np.array(timestamps, dtype=np.uint32)

np.savez("decoded.npz",
         timestamps=timestamps_array,
         gyro=gyro_array,
         accel=accel_array,
         mag=mag_array)

print("shranjeni!!")

# Preveri izgubljene pakete (gaps v packet_counter)
#counters = [p['packet_counter'] for p in packets]
#for i in range(1, len(counters)):
 #   expected = (counters[i-1] + 1) % 254
 #   if counters[i] != expected:
#       print(f"Izgubljen paket med {counters[i-1]} in {counters[i]}")

print(f"ziroskop vzorci: {len(gyro_samples)}")
print(f"accel vzorcei: {len(accel_samples)}")
print(f"magnetometer vzorci: {len(mag_sampples)}")
