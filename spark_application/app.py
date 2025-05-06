import argparse
import json
import os
import threading
import time
from pathlib import Path

import psutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col

RESULTS_PATH = Path('/opt/results.jsonl')


class MemoryMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.memory_usage_series = []
        self._stop_event = threading.Event()
        self.start_time = None

    def run(self):
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            elapsed = time.time() - self.start_time
            mem_usage_mb = process.memory_info().rss / (1024 * 1024)
            self.memory_usage_series.append((elapsed, mem_usage_mb))
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

    def get_memory_usage(self):
        return self.memory_usage_series

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.join()


def main():
    parser = argparse.ArgumentParser(
        description="Spark Application for checking performance with the mushrooms dataset with different settings"
    )
    parser.add_argument("--optimizations_enabled", action="store_true",
                        help="Apply cache, repartition")
    parser.add_argument("--datanodes_amount", type=int, default=1,
                        help="Number of DataNodes that is being used")
    args = parser.parse_args()

    optimizations_enabled = args.optimizations_enabled
    datanotes_amount = args.datanodes_amount

    spark = SparkSession.builder \
        .appName("MushroomsSparkExperiment") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.cores", "1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print("Spark session created.")
    print(f"Settings: optimizations_enabled={optimizations_enabled}, datanotes_amount={datanotes_amount}")

    with MemoryMonitor(interval=1.0) as monitor:
        print("Reading data from mushrooms.csv...")
        df = (spark.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv("hdfs://namenode:9000/user/hadoop/mushrooms.csv"))

        if optimizations_enabled:
            print("Applying optimizations: repartition and cache")
            df = df.repartition(4)
            df.cache()
            df.count()  # Trigger the cache
        else:
            print("No optimizations applied.")

        actions_start_time = time.time()

        count_initial = df.count()
        print(f"Total number of rows in the dataset: {count_initial}")

        print("Performing aggregation: count by 'class'...")
        agg_class_df = df.groupBy("class").count()
        agg_class_results = agg_class_df.collect()
        print("Aggregation results (class count):")
        for row in agg_class_results:
            print(row)

        print("Sorting aggregated results by count descending...")
        sorted_class_df = agg_class_df.orderBy("count", ascending=False)
        sorted_class_results = sorted_class_df.collect()
        print("Top aggregated results:")
        for row in sorted_class_results[:5]:
            print(row)

        numeric_cols = ["cap-diameter", "stem-height", "stem-width"]
        print("Calculating summary statistics for columns:", numeric_cols)
        summary_df = df.select(*[col(c) for c in numeric_cols])
        summary_stats = summary_df.describe().collect()
        print("Summary statistics:")
        for row in summary_stats:
            print(row)

        print("Filtering rows where cap-color == 'w' and calculating average cap-diameter...")
        filtered_df = df.filter(col("cap-color") == "w")
        avg_cap_diameter = filtered_df.agg(avg("cap-diameter").alias("avg_cap_diameter")
                                           ).collect()[0]["avg_cap_diameter"]
        print("Average cap-diameter for mushrooms with cap-color 'w':", avg_cap_diameter)

        print("Adding a new column 'height_width_ratio' (stem-height / stem-width)...")
        df_with_ratio = df.withColumn("height_width_ratio", col("stem-height") / col("stem-width"))
        df_with_ratio.select("stem-height", "stem-width", "height_width_ratio").show(10)

        end_time = time.time()
        actions_time = end_time - actions_start_time
        total_time = end_time - monitor.start_time
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Actions time: {actions_time:.2f} seconds")

        memory_usage_series = monitor.get_memory_usage()

    result = {
        "datanotes_amount": datanotes_amount,
        "optimizations_enabled": optimizations_enabled,
        "actions_time": actions_time,
        "total_time": total_time,
        "memory": memory_usage_series
    }

    with RESULTS_PATH.open('a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(result) + "\n")
    print("Experiment result appended to the JSONL file!")

    spark.stop()


if __name__ == "__main__":
    main()
