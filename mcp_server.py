"""
QBM Agent MCP Server (Local-Only Version)
-----------------------------------------

This MCP server exposes a Quantum Behavior Manager (QBM) that runs
*entirely on local Qiskit + Aer simulators* — no IBM Cloud account,
no tokens, no Runtime service.

Deps in your venv:
    pip install qiskit qiskit-aer "mcp[cli]"
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional

import json
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as Estimator  # Local Aer estimator

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
#  QBM core data models
# ---------------------------------------------------------------------------

@dataclass
class QBMRun:
    run_id: str
    best_params: List[float]
    best_value: float
    history: List[Dict[str, Any]]


@dataclass
class QBMTask:
    task_id: str
    description: str
    problem_spec: Dict[str, Any]
    ansatz_spec: Dict[str, Any]
    backend_prefs: Dict[str, Any]
    optimizer_spec: Dict[str, Any]
    runs: Dict[str, QBMRun] = field(default_factory=dict)


class QuantumBehaviorManager:
    """
    Minimal Quantum Behavior Manager for VQE-style tasks,
    using ONLY local Qiskit Aer simulators (no cloud).
    """

    def __init__(self) -> None:
        # In the local version we don't need a service or account.
        self.tasks: Dict[str, QBMTask] = {}

    # ------------------------------------------------------------------
    # Task handling
    # ------------------------------------------------------------------
    def define_task(self, data: Dict[str, Any]) -> QBMTask:
        task = QBMTask(
            task_id=data["task_id"],
            description=data.get("description", ""),
            problem_spec=data["problem_spec"],
            ansatz_spec=data["ansatz_spec"],
            backend_prefs=data["backend_preferences"],
            optimizer_spec=data["optimizer_spec"],
        )
        self.tasks[task.task_id] = task
        return task

    # ------------------------------------------------------------------
    # Problem / ansatz construction
    # ------------------------------------------------------------------
    def build_hamiltonian(self, problem_spec: Dict[str, Any]) -> SparsePauliOp:
        """
        Build a SparsePauliOp from a list of [label, coeff] pairs.
        Example:
            "hamiltonian_paulis": [
                ["II", -1.0523732],
                ["ZI", 0.3979374],
                ...
            ]
        """
        paulis = [
            (label, coeff) for (label, coeff) in problem_spec["hamiltonian_paulis"]
        ]
        return SparsePauliOp.from_list(paulis)

    def build_ansatz(
        self,
        ansatz_spec: Dict[str, Any],
        num_qubits: int,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Simple Ry + CX chain ansatz with L layers.
        """
        layers = ansatz_spec.get("num_layers", 1)

        num_params = num_qubits * layers
        params = ParameterVector("θ", num_params)

        qc = QuantumCircuit(num_qubits)
        idx = 0
        for _ in range(layers):
            # Single-qubit Ry rotations
            for q in range(num_qubits):
                qc.ry(params[idx], q)
                idx += 1
            # Entangling CX chain
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)

        return qc, params

    # ------------------------------------------------------------------
    # Local VQE control loop using Aer EstimatorV2
    # ------------------------------------------------------------------
    def run_vqe(self, task: QBMTask) -> QBMRun:
        """
        Coordinate-descent style VQE over ansatz parameters,
        evaluated with local Aer EstimatorV2.
        """
        # Local Aer estimator: no backend, no account, no runtime
        estimator = Estimator()  # uses local simulator internally

        problem_spec = task.problem_spec
        h = self.build_hamiltonian(problem_spec)
        num_qubits = problem_spec["num_qubits"]
        ansatz, param_vec = self.build_ansatz(task.ansatz_spec, num_qubits)

        opt = task.optimizer_spec
        max_iters = opt.get("max_iters", 20)
        step_size = opt.get("step_size", 0.2)

        initial_params = opt.get("initial_params")
        if initial_params is None:
            initial_params = [0.1] * len(param_vec)

        best = np.array(initial_params, dtype=float)

        # Helper: evaluate expectation value at given parameters
        def eval_params(x: np.ndarray) -> tuple[float, Dict[str, Any]]:
            # Aer EstimatorV2 expects pubs like (circuit, observable, params)
            pub = (ansatz, h, x.tolist())
            job = estimator.run([pub])
            result = job.result()
            pub_result = result[0]
            value = float(pub_result.data.evs)  # scalar expectation value
            metadata = pub_result.metadata
            return value, metadata

        best_value, meta = eval_params(best)
        history: List[Dict[str, Any]] = [
            {"params": best.tolist(), "value": best_value, "metadata": meta}
        ]

        # Simple coordinate-descent exploration
        for _ in range(max_iters):
            improved = False
            for i in range(len(best)):
                for delta in (step_size, -step_size):
                    cand = best.copy()
                    cand[i] += delta

                    val, meta = eval_params(cand)
                    history.append(
                        {
                            "params": cand.tolist(),
                            "value": val,
                            "metadata": meta,
                        }
                    )
                    if val < best_value:
                        best_value = val
                        best = cand
                        improved = True

            if not improved:
                step_size *= 0.5
                if step_size < 1e-3:
                    break

        run_id = f"run_{task.task_id}"
        run = QBMRun(
            run_id=run_id,
            best_params=best.tolist(),
            best_value=best_value,
            history=history,
        )
        task.runs[run_id] = run
        return run


# Single global QBM instance for the MCP server
qbm = QuantumBehaviorManager()

# Create the MCP server
mcp = FastMCP("QBM Agent (Local)", json_response=True)


# ---------------------------------------------------------------------------
#  MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def qbm_list_backends() -> Dict[str, Any]:
    """
    In the local-only version, we just advertise the Aer-based Estimator
    as a single logical backend.
    """
    return {
        "backends": [
            {
                "name": "aer_estimator_v2_local",
                "num_qubits": None,  # depends on your circuit
                "simulator": True,
                "operational": True,
                "pending_jobs": 0,
                "description": "Local Qiskit Aer EstimatorV2 simulator backend",
            }
        ]
    }


@mcp.tool()
def qbm_define_task(
    task_id: str,
    description: str,
    problem_spec: Union[str, Dict[str, Any]],
    ansatz_spec: Union[str, Dict[str, Any]],
    backend_preferences: Union[str, Dict[str, Any]],
    optimizer_spec: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    
    # Normalize JSON-like parameters: accept either dicts or JSON strings
    if isinstance(problem_spec, str):
        problem_spec = json.loads(problem_spec)
    if isinstance(ansatz_spec, str):
        ansatz_spec = json.loads(ansatz_spec)
    if isinstance(backend_preferences, str):
        backend_preferences = json.loads(backend_preferences)
    if isinstance(optimizer_spec, str):
        optimizer_spec = json.loads(optimizer_spec)

    """
    Define and register a VQE-style QBM task.
    """
    data = {
        "task_id": task_id,
        "description": description,
        "problem_spec": problem_spec,
        "ansatz_spec": ansatz_spec,
        "backend_preferences": backend_preferences,
        "optimizer_spec": optimizer_spec,
    }
    task = qbm.define_task(data)

    return {
        "task_id": task.task_id,
        "status": "defined",
        "description": task.description,
    }


@mcp.tool()
def qbm_run_task(task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the VQE loop for a previously defined task on the local Aer backend.

    If task_id is omitted and exactly one task exists, that task is used.
    """
    # If no task_id was provided, and exactly one task exists, use that.
    if task_id is None:
        if not qbm.tasks:
            raise ValueError("No tasks defined yet. Define a task first via qbm_define_task.")
        if len(qbm.tasks) > 1:
            raise ValueError(
                "Multiple tasks exist. Please specify task_id explicitly."
            )
        # Take the only defined task
        task_id = next(iter(qbm.tasks.keys()))

    if task_id not in qbm.tasks:
        raise ValueError(f"Unknown task_id: {task_id}")

    task = qbm.tasks[task_id]
    run = qbm.run_vqe(task)

    return {
        "task_id": task_id,
        "run_id": run.run_id,
        "best_params": run.best_params,
        "best_value": run.best_value,
        "num_evals": len(run.history),
    }


@mcp.tool()
def qbm_get_run_history(task_id: str, run_id: str) -> Dict[str, Any]:
    """
    Retrieve the full optimization trace for a given task/run.
    """
    task = qbm.tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id: {task_id}")

    run = task.runs.get(run_id)
    if run is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    return {
        "task_id": task_id,
        "run_id": run_id,
        "history": run.history,
    }

@mcp.tool()
def qbm_define_default_h2() -> Dict[str, Any]:
    """
    Convenience tool: define a default 2-qubit H2 VQE task
    without needing to pass any JSON arguments from the client.
    """
    task_id = "h2_vqe_local_001"
    description = "Default local VQE task for 2-qubit H2 Hamiltonian (Aer)"

    problem_spec = {
        "type": "hamiltonian_minimization",
        "num_qubits": 2,
        "hamiltonian_paulis": [
            ["II", -1.0523732],
            ["ZI", 0.3979374],
            ["IZ", -0.3979374],
            ["ZZ", -0.0112801],
            ["XX", 0.1809312],
        ],
    }

    ansatz_spec = {
        "type": "ry_cx_chain",
        "num_layers": 2,
    }

    backend_preferences = {
        "min_num_qubits": 2,
        "allow_simulator": True,
        "preferred_backend": "aer_estimator_v2_local",
    }

    optimizer_spec = {
        "type": "coordinate_descent",
        "max_iters": 10,
        "initial_params": [0.1, 0.2, 0.3, 0.4],
        "step_size": 0.2,
    }

    data = {
        "task_id": task_id,
        "description": description,
        "problem_spec": problem_spec,
        "ansatz_spec": ansatz_spec,
        "backend_preferences": backend_preferences,
        "optimizer_spec": optimizer_spec,
    }

    task = qbm.define_task(data)

    return {
        "task_id": task.task_id,
        "status": "defined",
        "description": task.description,
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")