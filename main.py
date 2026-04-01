import struct
import serial
import serial.tools.list_ports
import sys

# v device maneger preveris in popravis za svoj racunalnik
SERIAL_PORT = "COM7"
BAUD_RATE   = 115200
# za tof se ni navodila ampak mislim da ima 0x05 oznako
CHUNK_NAMES = {0x01: "gyro", 0x02: "accel", 0x03: "mag", 0x05: "tof"}

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

def parse_packet(raw):
    if len(raw) < 5 or raw[0:2] != b'\xFF\xFF':
        return None

    packet_counter = raw[2]
    payload = unstuff_bytes(raw[3:])

    if len(payload) < 8:
        return None

    received_crc = struct.unpack('<H', payload[-2:])[0]
    computed_crc = crc16_compute(payload[:-2])

    if received_crc != computed_crc:
        print(f"  [!] CRC mismatch (got {received_crc:#06x}, expected {computed_crc:#06x})")
        print(f"      raw tail: {raw[-4:].hex()}")
        return None

    timestamp = struct.unpack('<I', payload[0:4])[0]
    
    chunks_data = payload[6:-2]
    pos = 0
    chunks = {}

    while pos + 4 <= len(chunks_data):
        chunk_id       = chunks_data[pos]
        chunk_size     = struct.unpack('<H', chunks_data[pos+1:pos+3])[0] + 1
        pos += 4

        if pos + chunk_size > len(chunks_data):
            break

        chunk_data = chunks_data[pos:pos + chunk_size]
        pos += chunk_size


        samples = []
        for i in range(0, len(chunk_data) - 5, 6):
            x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
            samples.append((x, y, z))

        chunks[chunk_id] = samples

        
        print(f"  chunk {chunk_id} raw bytes: {chunk_data.hex()}")

    return {"timestamp": timestamp, "packet_counter": packet_counter, "chunks": chunks}

def find_sync(buf, start=0):
    i = start
    while i < len(buf) - 1:
        if buf[i] == 0xFF and buf[i+1] == 0xFF:
            return i
        i += 1
    return -1

def main():
    print(f"Opening {SERIAL_PORT} @ {BAUD_RATE} baud ...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open port: {e}")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device} — {p.description}")
        sys.exit(1)

    buf = bytearray()
    last_counter = None

    try:
        while True:
            chunk = ser.read(256)
            if chunk:
                buf.extend(chunk)

            s1 = find_sync(buf)
            if s1 == -1:
                buf.clear()
                continue
            if s1 > 0:
                buf = buf[s1:]

            s2 = find_sync(buf, 2)
            if s2 == -1:
                if len(buf) > 4096:
                    buf.clear()
                continue

            raw = bytes(buf[:s2])
            buf = buf[s2:]

            while raw and raw[-1] in (0x0a, 0x0d):
                raw = raw[:-1]
            
            p_idx = raw.find(b'Packet:')
            if p_idx != -1:
                raw = raw[:p_idx]

            pkt = parse_packet(raw)
            if pkt is None:
                print(f"full raw: {raw.hex()}")
                continue

            cnt = pkt["packet_counter"]
            ts  = pkt["timestamp"]

            lost = ""
            if last_counter is not None:
                expected = (last_counter + 1) % 254
                if cnt != expected:
                    lost = f"  *** {(cnt - last_counter - 1) % 254} LOST ***"
            last_counter = cnt

            print(f"\n[#{cnt:3d}] t={ts} ms{lost}")
            for chunk_id, samples in pkt["chunks"].items():
                name = CHUNK_NAMES.get(chunk_id, f"chunk_{chunk_id:#04x}")
                for i, (x, y, z) in enumerate(samples):
                    print(f"  {name}[{i}]  x={x:6d}  y={y:6d}  z={z:6d}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()