import struct
import csv

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
            i+=1
            unstuffed.append(0xFE ^ data[i])
        else:
            unstuffed.append(data[i])
    return bytes(unstuffed)

def parse_packet(data):
    # 1. Find sync marker (0xFFFF)
    if data[0:2] != b'\xFF\xFF':
        return None
    
    # 2. Extract and unstuff payload
    payload = unstuff_bytes(data[2:])
    
    # 3. Parse header
    timestamp = struct.unpack('<I', payload[0:4])[0]
    packet_size_enc = struct.unpack('<H', payload[4:6])[0]
    
    # 4. Extract and verify CRC
    received_crc = struct.unpack('<H', payload[-2:])[0]
    computed_crc = crc16_compute(payload[:-2])
    if received_crc != computed_crc:
        return None  # CRC error
    
    # 5. Parse chunks
    chunks_data = payload[6:-2]
    pos = 0
    chunks = {}
    
    while pos < len(chunks_data):
        chunk_id = chunks_data[pos]
        chunk_size_enc = struct.unpack('<H', chunks_data[pos+1:pos+3])[0]
        chunk_size = chunk_size_enc + 1
        reserved = chunks_data[pos+3]
        chunk_data = chunks_data[pos+4:pos+4+chunk_size]
        
        # Parse sensor samples (3x int16_t per sample)
        samples = []
        for i in range(0, chunk_size, 6):
            x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
            samples.append((x, y, z))
        
        chunks[chunk_id] = samples
        pos += 4 + chunk_size
    
    return {'timestamp': timestamp, 'chunks': chunks}
