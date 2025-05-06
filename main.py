import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

from constants import ROOT_DIR


def run_command(command, capture_output=False, check=True):
    print(f"Executing: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        print("Error executing command:")
        print(e)
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error output:", e.stderr)
        sys.exit(1)


def wait_for_container_health(container_name, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        status = run_command(
            f"docker inspect --format='{{{{.State.Health.Status}}}}' {container_name}",
            capture_output=True,
            check=False
        )
        if status == "healthy":
            print(f"Container {container_name} is healthy.")
            return
        else:
            print(f"Waiting for {container_name} to be healthy... Current status: {status}")
            time.sleep(5)
    raise TimeoutError(f"{container_name} did not become healthy in {timeout} seconds.")


def upload_data_to_hdfs():
    print("Uploading data to HDFS...")
    run_command("docker exec namenode hdfs dfs -mkdir -p /user/hadoop")
    run_command("docker exec namenode hdfs dfs -put -f /hadoop/mushrooms.csv /user/hadoop/mushrooms.csv")


def run_spark_job(optimizations_enabled, datanodes_amount):
    print(f"Running Spark job with settings: optimizations_enabled={optimizations_enabled}, "
          f"datanodes_amount={datanodes_amount}")
    command = (
        f"docker exec spark /spark/bin/spark-submit --master spark://spark:7077 "
        f"/opt/spark_application/app.py --datanodes_amount {datanodes_amount}"
    )
    if optimizations_enabled:
        command += " --optimizations_enabled"
    run_command(command)

def generate_plots():
    file_path = ROOT_DIR / 'results.jsonl'
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    records = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line}\n{e}")

    if not records:
        print("No experiment records found in the results file.")
        return

    records = sorted(records, key=lambda r: (r['datanotes_amount'], r['optimizations_enabled']))

    labels = []
    actions_times = []
    total_times = []
    for rec in records:
        label = f"{rec['datanotes_amount']} DataNode{'s' if rec['datanotes_amount'] > 1 else ''} - " + \
                ("Optimized" if rec['optimizations_enabled'] else "Unoptimized")
        labels.append(label)
        actions_times.append(rec["actions_time"])
        total_times.append(rec["total_time"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    axes[0].bar(labels, actions_times)
    axes[0].set_xlabel("Experiment Settings")
    axes[0].set_ylabel("Actions Time (sec)")
    axes[0].set_title("Execution Time for Spark Actions")

    axes[1].bar(labels, total_times)
    axes[1].set_xlabel("Experiment Settings")
    axes[1].set_ylabel("Total Time (sec)")
    axes[1].set_title("Total Execution Time (including initialization)")

    for rec, label in zip(records, labels):
        memory_series = rec.get("memory", [])
        if memory_series:
            times, mem_values = zip(*memory_series)
            axes[2].plot(times, mem_values, label=label)
        else:
            print(f"No memory usage data for {label}")

    axes[2].set_xlabel("Time (sec)")
    axes[2].set_ylabel("Memory Usage (MB)")
    axes[2].set_title("Memory Usage Over Time")
    axes[2].legend(title="Experiment Settings", loc='best')

    plt.tight_layout()
    save_path = ROOT_DIR / "plot.png"
    plt.savefig(save_path)
    plt.close()

def main():
    if Path(ROOT_DIR / "results.jsonl").exists():
        Path(ROOT_DIR / "results.jsonl").unlink()

    print("Rebuilding Docker containers...")
    run_command("docker-compose build --no-cache")

    # 1 DataNode.
    print("Launching cluster with 1 DataNode...")
    run_command("docker-compose up -d --scale datanode=1")
    print("Waiting for Namenode to become healthy...")
    wait_for_container_health("namenode")
    upload_data_to_hdfs()

    print("Experiment 1: 1 DataNode, unoptimized")
    run_spark_job(optimizations_enabled=False, datanodes_amount=1)
    print("Experiment 2: 1 DataNode, optimized")
    run_spark_job(optimizations_enabled=True, datanodes_amount=1)

    print("Stopping the cluster...")
    run_command("docker-compose down")

    # 3 DataNodes.
    print("Launching cluster with 3 DataNodes...")
    run_command("docker-compose up -d --scale datanode=3")
    print("Waiting for Namenode to become healthy...")
    wait_for_container_health("namenode")
    upload_data_to_hdfs()

    print("Experiment 3: 3 DataNodes, unoptimized")
    run_spark_job(optimizations_enabled=False, datanodes_amount=3)
    print("Experiment 4: 3 DataNodes, optimized")
    run_spark_job(optimizations_enabled=True, datanodes_amount=3)

    print("Stopping the cluster...")
    run_command("docker-compose down")

    print("All settings run successfully!")

    # Generate plot
    print("Generating the plot...")
    generate_plots()
    print("Plot generated successfully!")

if __name__ == "__main__":
    main()
