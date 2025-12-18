# qiskit_mcp — QBM Agent Prototype

Prototype files for a Quantum Behavior Manager (QBM) MCP server and basic test scripts.

## Files
- `mcp_server.py` — MCP server entrypoint exposing QBM-related tools
- `configure_ibm_account.py` — IBM account configuration helper
- `test_qbm_core_local.py` — local QBM core tests
- `test_ibm_service.py` — IBM service connectivity tests

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python mcp_server.py
