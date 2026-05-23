from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform")
job = service.job("d85244voha1c73bobcag")
job.cancel()
print("Job cancelled")