import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from numpy.lib.shape_base import expand_dims

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (GaussianEmitter, ImprovementEmitter, IsoLineEmitter,
                           OptimizingEmitter, RandomDirectionEmitter)
from ribs.optimizers import Optimizer
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap

def sty(dims, sol):
    scores = 0
    if sol.ndim == 1:
        # if its a 1d
        for i in range(dims):
            scores = scores + ( np.power(sol[i], 4) - 16 * np.power(sol[i], 2) + 5 * sol[i] )
        scores = 0.5 * scores
        return scores

    else:
        for i in range(dims):
            scores = scores + ( np.power(sol[:, i], 4) - 16 * np.power(sol[:, i], 2) + 5 * sol[:, i] )
        scores = 0.5 * scores
        return scores

def tang(sol):
    """Sphere function evaluation and BCs for a batch of solutions.
    Args:
        sol (np.ndarray): (batch_size, dim) array of solutions
    Returns:
        objs (np.ndarray): (batch_size,) array of objective values
        bcs (np.ndarray): (batch_size, 2) array of behavior values
    """
    dims = sol.shape[1]

    best = np.zeros(dims) - 2.903534
    best_obj = sty(dims, best)

    worst_obj = (.5*5**4 - 8*5**2 + 2.5*5) * dims
    raw_obj = sty(dims, sol)
    objs = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Calculate BCs.
    clipped = sol.copy()
    clip_indices = np.where(np.logical_or(clipped > 5, clipped < -5))
    clipped[clip_indices] = 5 / clipped[clip_indices]
    bcs = np.concatenate(
        (
            np.sum(clipped[:, :dims // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dims // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objs, bcs


def create_optimizer(algorithm, dim, seed):
    """Creates an optimizer based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        dim (int): Dimensionality of the sphere function.
        seed (int): Main seed or the various components.
    Returns:
        Optimizer: A ribs Optimizer for running the algorithm.
    """
    max_bound = dim / 2 * 5.12
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    initial_sol = np.zeros(dim)
    batch_size = 37
    num_emitters = 15

    # Create archive.
    if algorithm in [
            "map_elites", "line_map_elites", "cma_me_imp", "cma_me_imp_mu",
            "cma_me_rd", "cma_me_rd_mu", "cma_me_opt", "cma_me_mixed"
    ]:
        archive = GridArchive((500, 500), bounds, seed=seed)
    elif algorithm in ["cvt_map_elites", "line_cvt_map_elites"]:
        archive = CVTArchive(10_000, bounds, samples=100_000, use_kd_tree=True)
    else:
        raise ValueError(f"Algorithm `{algorithm}` is not recognized")

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else list(
        range(seed, seed + num_emitters))
    if algorithm in ["map_elites", "cvt_map_elites"]:
        emitters = [
            GaussianEmitter(archive,
                            initial_sol,
                            0.5,
                            batch_size=batch_size,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["line_map_elites", "line_cvt_map_elites"]:
        emitters = [
            IsoLineEmitter(archive,
                           initial_sol,
                           iso_sigma=0.1,
                           line_sigma=0.2,
                           batch_size=batch_size,
                           seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_imp", "cma_me_imp_mu"]:
        selection_rule = "filter" if algorithm == "cma_me_imp" else "mu"
        emitters = [
            ImprovementEmitter(archive,
                               initial_sol,
                               0.5,
                               batch_size=batch_size,
                               selection_rule=selection_rule,
                               seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_rd", "cma_me_rd_mu"]:
        selection_rule = "filter" if algorithm == "cma_me_rd" else "mu"
        emitters = [
            RandomDirectionEmitter(archive,
                                   initial_sol,
                                   0.5,
                                   batch_size=batch_size,
                                   selection_rule=selection_rule,
                                   seed=s) for s in emitter_seeds
        ]
    elif algorithm == "cma_me_opt":
        emitters = [
            OptimizingEmitter(archive,
                              initial_sol,
                              0.5,
                              batch_size=batch_size,
                              seed=s) for s in emitter_seeds
        ]
    elif algorithm == "cma_me_mixed":
        emitters = [
            RandomDirectionEmitter(
                archive, initial_sol, 0.5, batch_size=batch_size, seed=s)
            for s in emitter_seeds[:7]
        ] + [
            ImprovementEmitter(
                archive, initial_sol, 0.5, batch_size=batch_size, seed=s)
            for s in emitter_seeds[7:]
        ]

    return Optimizer(archive, emitters)


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    if isinstance(archive, GridArchive):
        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, vmin=0, vmax=100)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    elif isinstance(archive, CVTArchive):
        plt.figure(figsize=(16, 12))
        cvt_archive_heatmap(archive, vmin=0, vmax=100)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def Styblinski_main(algorithm,
                dim=20,
                itrs=4500,
                outdir="Styblinski_output",
                log_freq=250,
                seed=None):
    """Demo on the Sphere function.

    Args:
        algorithm (str): Name of the algorithm.
        dim (int): Dimensionality of solutions.
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations to wait before recording metrics
            and saving heatmap.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
    name = f"{algorithm}_{dim}"
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()

    optimizer = create_optimizer(algorithm, dim, seed)
    archive = optimizer.archive
    metrics = {
        "QD Score": {
            "x": [0],
            "y": [0.0],
        },
        "Archive Coverage": {
            "x": [0],
            "y": [0.0],
        },
    }

    non_logging_time = 0.0
    with alive_bar(itrs) as progress:
        save_heatmap(archive, str(outdir / f"{name}_heatmap_{0:05d}.png"))

        for itr in range(1, itrs + 1):
            itr_start = time.time()
            sols = optimizer.ask()
            objs, bcs = tang(sols)
            optimizer.tell(objs, bcs)
            non_logging_time += time.time() - itr_start
            progress()

            # Logging and output.
            final_itr = itr == itrs
            if itr % log_freq == 0 or final_itr:
                data = archive.as_pandas(include_solutions=final_itr)
                if final_itr:
                    data.to_csv(str(outdir / f"{name}_archive.csv"))

                # Record and display metrics.
                total_cells = 10_000 if isinstance(archive,
                                                   CVTArchive) else 500 * 500
                metrics["QD Score"]["x"].append(itr)
                metrics["QD Score"]["y"].append(data['objective'].sum())
                metrics["Archive Coverage"]["x"].append(itr)
                metrics["Archive Coverage"]["y"].append(
                    len(data) / total_cells * 100)
                print(f"Iteration {itr} | Archive Coverage: "
                      f"{metrics['Archive Coverage']['y'][-1]:.3f}% "
                      f"QD Score: {metrics['QD Score']['y'][-1]:.3f}")

                save_heatmap(archive,
                             str(outdir / f"{name}_heatmap_{itr:05d}.png"))

    # Plot metrics.
    print(f"Algorithm Time (Excludes Logging and Setup): {non_logging_time}s")
    for metric in metrics:
        plt.plot(metrics[metric]["x"], metrics[metric]["y"])
        plt.title(metric)
        plt.xlabel("Iteration")
        plt.savefig(
            str(outdir / f"{name}_{metric.lower().replace(' ', '_')}.png"))
        plt.clf()
    with (outdir / f"{name}_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


if __name__ == '__main__':
    fire.Fire(Styblinski_main)