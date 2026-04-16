import serial
import time
import re
import os

PORT = "COM7"
BAUD = 115200
SAVE_DIR = "logs"

os.makedirs(SAVE_DIR, exist_ok=True)


def read_until_prompt(ser, timeout=2):
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
    ser.reset_input_buffer()
    ser.write(b"LIST\n")
    time.sleep(0.5)

    output = read_until_prompt(ser)

    files = re.findall(r"(LOG\d+\.BIN)", output)
    return files


def download_file(ser, filename):
    print(f"\n⬇ Downloading {filename} ...")

    ser.reset_input_buffer()
    ser.write(f"GET {filename}\n".encode())

    path = os.path.join(SAVE_DIR, filename)

    with open(path, "wb") as f:
        last = time.time()

        while True:
            data = ser.read(1024)

            if data:
                f.write(data)
                last = time.time()
            else:
                if time.time() - last > 2:
                    break

    print(f"✔ Saved: {path}")


def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.5)
    time.sleep(2)

    files = get_file_list(ser)

    if not files:
        print("No files found!")
        return

    print("\nAvailable files:")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")

    choice = input("\nEnter index or filename: ").strip()

    if choice.isdigit():
        idx = int(choice)
        if idx < 0 or idx >= len(files):
            print("Invalid index")
            return
        filename = files[idx]
    else:
        if choice not in files:
            print("File not found")
            return
        filename = choice

    download_file(ser, filename)

    ser.close()
    print("\nDONE")


if __name__ == "__main__":
    main()