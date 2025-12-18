# configure_ibm_account.py

from qiskit_ibm_runtime import QiskitRuntimeService

# 1) Remove any previously saved (possibly wrong) account
try:
    deleted = QiskitRuntimeService.delete_account()
    print("Deleted existing account config:", deleted)
except Exception as e:
    print("No existing account or deletion issue:", e)

#2 configure_ibm_account.py
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="qbm_agent: 0gpNDwNLhdDvgdDNDVPqV6XkOdvO71JcGaXkfzvP4D3x",
    # instance="crn:v1:bluemix:public:quantum-computing:us-east:a/f13ad245639d481887907cef24848ee5:f85c25de-a794-4aa6-9022-bf7d183fa43e::",
    set_as_default=True,
    overwrite=True,
)