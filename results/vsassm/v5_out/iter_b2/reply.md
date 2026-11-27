```python
def isolated_store_worker(script_path, store_data, memory, budget, p, design,
                          harder_p):
    """Import the standalone script as a real module inside each worker."""
    import importlib.util
    import sys

    module_name = "_part_b_worker_implementation"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load simulation script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        # Registration must precede execution for dataclasses and pickle.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise

    store = module.Store(**store_data)
    return module.evaluate_store(
        store, memory, budget, p, design, harder_p)


def pooled_evaluate(pool, stores, memory, budget, p, design, harder_p):
    # Send only a small, self-contained worker function and plain data.
    # Cached functions defined in __main__ otherwise get pickled by reference
    # and cannot be resolved by loky's worker bootstrap module.
    script_path = str(Path(__file__).resolve())
    rows = pool(delayed(isolated_store_worker)(
        script_path, dict(s.__dict__), memory, float(budget), p, design,
        harder_p) for s in stores)
    exact = sum(row["exact"] for row in rows)
    probes = sum(row["probes"] for row in rows)
    return {
        "exact": exact,
        "probes": probes,
        "share": exact / probes,
        "fits": all(row.get("fits", True) for row in rows),
        "failed_insertions": sum(row["failed_insertions"] for row in rows),
        "cells": [row["cells"] for row in rows],
    }
```

Workers now import the script under a stable module name instead of serializing its cached functions from `__main__`. Store data is sent as a plain dictionary and reconstructed inside each worker.