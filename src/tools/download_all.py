import serial
import time
import re
import os

PORT = "COM7"
BAUD = 115200
SAVE_DIR = "logs"

os.makedirs(SAVE_DIR, exist_ok=True)


def read_until_prompt(ser, timeout=2):
    """Preberi dokler ne vidiš '>' prompta"""
    end_time = time.time() + timeout
    data = b""

    while time.time() < end_time:
        chunk = ser.read(1024)
        if chunk:
            data += chunk
            if b'>' in data:
                break

    return data.decode(errors="ignore")


def get_file_list(ser):
    ser.write(b"LIST\n")
    time.sleep(0.5)

    output = read_until_prompt(ser)
    print("LIST output:\n", output)

    files = re.findall(r"(LOG\d+\.BIN)", output)
    return files


def download_file(ser, filename):
    print(f"\n⬇ Downloading {filename} ...")

    ser.reset_input_buffer()
    ser.write(f"GET {filename}\n".encode())

    path = os.path.join(SAVE_DIR, filename)

    with open(path, "wb") as f:
        last_data_time = time.time()

        while True:
            data = ser.read(1024)

            if data:
                f.write(data)
                last_data_time = time.time()
            else:
                if time.time() - last_data_time > 2:
                    break

    print(f"Saved: {path}")


def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.5)
    time.sleep(2)

    # reset state
    ser.write(b"\n")
    time.sleep(0.5)

    files = get_file_list(ser)

    print(f"\nNajdenih {len(files)} datotek")

    for f in files:
        download_file(ser, f)

    ser.close()
    print("\nDONE")


if __name__ == "__main__":
    main()