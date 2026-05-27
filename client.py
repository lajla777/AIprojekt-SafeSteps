import socket

s = socket.socket()
s.connect(("127.0.0.1", 5000))

while True:
    cmd = input("> ")   
    if cmd == "exit":
        break

    s.sendall((cmd + "\n").encode())
    print(s.recv(4096).decode())

s.close()
