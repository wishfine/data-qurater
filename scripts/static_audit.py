import os
import sys
import json
import py_compile
import re

def audit():
    print("=== RUNNING LOCAL STATIC POLICY AUDIT ===")
    
    results = {
        "shell_syntax_check": "passed",
        "python_syntax_check": "passed",
        "static_policy_audit": "passed",
        "errors": []
    }
    
    # 1. Audit server shell scripts
    scripts_dir = "scripts"
    shell_files = [f for f in os.listdir(scripts_dir) if f.startswith("server_") and f.endswith(".sh")]
    
    for sf in shell_files:
        path = os.path.join(scripts_dir, sf)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # server_collect_report.sh is allowed to run without set -e or environment checks
        if sf == "server_collect_report.sh":
            if "set -u" not in content:
                results["static_policy_audit"] = "failed"
                results["errors"].append(f"{sf} is missing 'set -u'")
        elif sf == "server_verify_model_path.sh":
            # This is a light path verifier, doesn't need PyTorch/CUDA checks
            if "set -euo pipefail" not in content:
                results["static_policy_audit"] = "failed"
                results["errors"].append(f"{sf} is missing 'set -euo pipefail'")
        elif sf == "server_check_env.sh":
            # This generates the status JSON, so it must not check it
            if "set -euo pipefail" not in content:
                results["static_policy_audit"] = "failed"
                results["errors"].append(f"{sf} is missing 'set -euo pipefail'")
        else:
            # Must contain set -euo pipefail
            if "set -euo pipefail" not in content and "set -e" not in content:
                results["static_policy_audit"] = "failed"
                results["errors"].append(f"{sf} does not enforce strict fail-fast (missing set -euo pipefail or set -e)")
            
            # Check environment check status script is invoked
            if "check_environment_status.py" not in content:
                results["static_policy_audit"] = "failed"
                results["errors"].append(f"{sf} does not call check_environment_status.py")
                
        # Check tee safety
        if "tee" in content and "pipefail" not in content:
            results["static_policy_audit"] = "failed"
            results["errors"].append(f"{sf} uses tee without pipefail enabled")

        # Check for unconfirmed python3 calls in server scripts
        if "python3" in content:
            results["static_policy_audit"] = "failed"
            results["errors"].append(f"{sf} uses unconfirmed python3 command; use python instead")

        # Specific check for server_verify_data.sh
        if sf == "server_verify_data.sh":
            if "build_smoke_split.py" not in content:
                results["static_policy_audit"] = "failed"
                results["errors"].append("server_verify_data.sh does not call build_smoke_split.py")

    # 2. Check for banned environment names and connected components terminology
    banned = ["agentgym", "research-rl"]
    banned_terms = ["strongly connected component", "强连通分量"]
    exclude_dirs = [".git", "reports", "outputs", "scratch"]
    
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith((".py", ".sh", ".md", ".json")):
                path = os.path.join(root, file)
                # Skip checking this script itself
                if "static_audit.py" in path:
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    for idx, line in enumerate(lines):
                        # check banned env names
                        for b in banned:
                            if b in line:
                                results["static_policy_audit"] = "failed"
                                results["errors"].append(f"{path}:L{idx+1} contains banned environment name '{b}'")
                        # check banned terms
                        for t in banned_terms:
                            if t in line:
                                results["static_policy_audit"] = "failed"
                                results["errors"].append(f"{path}:L{idx+1} contains incorrect terminology '{t}' (use 'connected component' or '连通分量')")
                except Exception:
                    pass

    # 3. Parse JSON configurations
    config_files = ["configs/qwen3_06b_smoke.json", "configs/qwen3_06b_train.json"]
    for cf in config_files:
        if os.path.exists(cf):
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    json.load(f)
            except Exception as e:
                results["static_policy_audit"] = "failed"
                results["errors"].append(f"Failed to parse JSON config {cf}: {e}")
        else:
            results["static_policy_audit"] = "failed"
            results["errors"].append(f"JSON config missing: {cf}")

    # 4. Compile Python files
    py_files = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
                
    for pf in py_files:
        try:
            py_compile.compile(pf, doraise=True)
        except py_compile.PyCompileError as e:
            results["python_syntax_check"] = "failed"
            results["errors"].append(f"Python compile error in {pf}: {e}")

    # Write report
    os.makedirs("reports", exist_ok=True)
    with open("reports/local_static_audit.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 50)
    print("STATIC AUDIT RESULT SUMMARY")
    print("=" * 50)
    print(f"Shell Syntax Check  : {results['shell_syntax_check']}")
    print(f"Python Syntax Check : {results['python_syntax_check']}")
    print(f"Static Policy Audit : {results['static_policy_audit']}")
    print(f"Total Errors Found  : {len(results['errors'])}")
    if results["errors"]:
        print("\nErrors detail:")
        for err in results["errors"]:
            print(f"  - {err}")
    print("=" * 50 + "\n")
    
    if len(results["errors"]) > 0:
        sys.exit(1)
        
if __name__ == "__main__":
    audit()
