import abc
import json
import time
import torch
import typing as t

import mirrordescent.distributions as dists
import mirrordescent.mirror_maps as mm
import mirrordescent.sampler as sampler


class Experiment(abc.ABC):
    """Base class for experiments."""
    def __init_subclass__(cls, **kwargs):
        ExperimentRegistry.REGISTRY.append(cls)

    @abc.abstractmethod
    def run(self) -> t.Any:
        """Some method to run and store the results of the experiment."""

class ExperimentRegistry:
    REGISTRY: list[t.Type[Experiment]] = []


class DimensionExperiment(Experiment):
    DIMS = [2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 96, 128]
    RUNS = 10
    NO_SAMPLES = 100

    RESULTS_FILE = './dimensions.json'

    @staticmethod
    def step_sizes(t: int) -> float:
        return t ** -1

    def _run_single(self, dim: int) -> float:
        dist = dists.DirichletPosterior(
            concentration=torch.ones(dim) * 0.1
        )

        mld = sampler.MLD(
            mirror_map=mm.EntropicMirrorMap(),
            step_sizes=self.step_sizes,
            dist=dist
        )

        start_time = time.perf_counter()
        _ = mld.get_samples(self.NO_SAMPLES)
        end_time = time.perf_counter()

        return end_time - start_time

    def run(self) -> dict[int, list[float]]:
        try:
            with open(self.RESULTS_FILE, 'r') as f:
                results: dict[int, list[float]] = json.load(f)
        except FileNotFoundError:
            results: dict[int, list[float]] = {}

        # For some godforsaken reason, the keys are all strings
        results = {int(k): v for k, v in results.items()}

        try:
            for dim in self.DIMS:
                results[dim] = results.get(dim, [])
                completed_runs = len(results[dim])
                print(f'Completed runs for dimension {dim}: {completed_runs}')
                if completed_runs == self.RUNS:
                    continue
                
                for run in range(completed_runs+1, self.RUNS+1):
                    print(f'Starting run {run} for dimension {dim}')
                    time_taken = self._run_single(dim)
                    print(f'\tCompleted in {time_taken:.2f} seconds')
                    results[dim].append(time_taken)
        except Exception:
            with open(self.RESULTS_FILE, 'w') as f:
                json.dump(results, f)
            raise
        finally:
            with open(self.RESULTS_FILE, 'w') as f:
                json.dump(results, f)
        
        return results
    

if __name__ == '__main__':
    for experiment in ExperimentRegistry.REGISTRY:
        experiment().run()