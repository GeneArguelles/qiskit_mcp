# test_qbm_core_local.py

from mcp_server import QuantumBehaviorManager

def main():
    qbm = QuantumBehaviorManager()

    task_data = {
        "task_id": "h2_vqe_local_001",
        "description": "Local VQE run on a simple 2-qubit H2 Hamiltonian (Aer)",
        "problem_spec": {
            "type": "hamiltonian_minimization",
            "num_qubits": 2,
            "hamiltonian_paulis": [
                ["II", -1.0523732],
                ["ZI", 0.3979374],
                ["IZ", -0.3979374],
                ["ZZ", -0.0112801],
                ["XX", 0.1809312],
            ],
        },
        "ansatz_spec": {
            "type": "ry_cx_chain",
            "num_layers": 2,
        },
        "backend_preferences": {
            # not really used in local mode, but kept for schema compatibility
            "min_num_qubits": 2,
            "allow_simulator": True,
            "preferred_backend": "aer_estimator_v2_local",
        },
        "optimizer_spec": {
            "type": "coordinate_descent",
            "max_iters": 10,
            "initial_params": [0.1, 0.2, 0.3, 0.4],
            "step_size": 0.2,
        },
    }

    task = qbm.define_task(task_data)
    run = qbm.run_vqe(task)

    print("✅ Local QBM VQE run completed.")
    print("Run ID:     ", run.run_id)
    print("Best value: ", run.best_value)
    print("Best params:", run.best_params)
    print("Evaluations:", len(run.history))

if __name__ == "__main__":
    main()