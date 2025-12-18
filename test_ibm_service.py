# test_ibm_service.py
from qiskit_ibm_runtime import QiskitRuntimeService

def main():
    service = QiskitRuntimeService()
    print("✅ QiskitRuntimeService initialized successfully.")
    print("Available backends:")
    for backend in service.backends():
        print("-", backend.name)

if __name__ == "__main__":
    main()