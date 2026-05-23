from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform")

# List available backends
backends = service.backends()
for b in backends:
    print(f"{b.name} — qubits: {b.num_qubits} — status: {b.status().status_msg}")

for name in ["ibm_kingston", "ibm_marrakesh", "ibm_fez"]:
    backend = service.backend(name)
    status = backend.status()
    print(f"{name} — jobs in queue: {status.pending_jobs}")