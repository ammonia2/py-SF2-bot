def check_port(port):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0  # Returns True if port is in use

# Add before connecting:
if check_port(10000):
    print("Warning: Port 10000 is already in use by another application!")
else:
    print("Port available")