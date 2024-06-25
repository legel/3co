import socket
import time

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serv.bind(('192.168.0.42', 8080))
serv.listen(5)

while True:
    conn, addr = serv.accept()
    from_client = ''
    while True:
        data = conn.recv(4096)
        if not data: break
        from_client += data.decode()
        print(from_client)
        time.sleep(5.0)
        conn.send("I am SERVER".encode())
    conn.close()
