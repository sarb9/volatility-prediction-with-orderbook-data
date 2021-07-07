import argparse

from experiment_engine.engine import ExperimentEngine

if __name__ == "__main__":
    experiment_engine: ExperimentEngine = ExperimentEngine()

    parser = argparse.ArgumentParser(description="Timeseries analysis by SARB.")

    parser.add_argument(
        "--run-experiment",
        "-r",
        help='Experiment file in yaml format in "experiments" folder.',
    )

    args = parser.parse_args()

    experiment_engine.run_experiment_file(args.run_experiment)
