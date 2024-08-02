import os
import logging
import subprocess

def get_latest_directory() -> str:
    """ Get the latest simulation results directory """

    commit_hash = get_latest_commit_hash()

    if not os.path.exists("sim_results"):
        raise FileNotFoundError("No sim_results directory found")
    
    commit_hash_dir = os.path.join("sim_results", commit_hash)

    if not os.path.exists(commit_hash_dir):
        raise FileNotFoundError("No simulation results found for latest commit")
    
    return commit_hash_dir

def get_latest_sim() -> str:
    """ Get the latest simulation results file """

    commit_hash_dir = get_latest_directory()
    
    # Get the sim file with the latest timestamp
    latest_file = max(
        os.listdir(commit_hash_dir),
        key=lambda x: os.path.getctime(os.path.join(commit_hash_dir, x))
    )

    return os.path.join(commit_hash_dir, latest_file)

def get_latest_commit_hash() -> str:
    """ Get the latest commit hash of the git repo """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        commit_hash = result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        logging.warning(f"No git repo found: {e.stderr.decode('utf-8')}")
        commit_hash = "no_repo_found"
    
    return commit_hash