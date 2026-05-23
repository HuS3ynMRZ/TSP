from qiskit_ibm_runtime import QiskitRuntimeService

# Read token from file
with open("IBM.txt", "r") as f:
    token = f.read().strip()

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=token,
    overwrite=True
)

print("Token saved successfully!")