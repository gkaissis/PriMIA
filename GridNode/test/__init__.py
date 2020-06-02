PORTS = ["3000", "3001", "3002", "3003"]
IDS = ["bob", "alice", "james", "dan"]

worker_ports = {node_id: port for node_id, port in zip(IDS, PORTS)}
