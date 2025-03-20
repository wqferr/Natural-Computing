import numpy as np
from matplotlib import pyplot as plt


def objective_function(x: float) -> float:
    return 2 ** (-2 * ((x - 0.1) / 0.9) ** 2) * np.sin(5 * np.pi * x) ** 6


def perturb(x: float, rng: np.random.Generator, perturb_scale: float = 0.03) -> float:
    return x + rng.normal(0, perturb_scale)


def hillclimbing(
    *, max_iter: int = 1000, max_iter_without_improvement: int = 100, rng_seed: int = 1
):
    rng = np.random.default_rng(rng_seed)
    progression: list[float] = []
    progression.append(rng.normal(0.5, 0.25))
    num_iter = 0
    current_eval = objective_function(progression[-1])
    num_iter_without_improvement = 0

    while num_iter < max_iter:
        new_x = perturb(progression[-1], rng)
        new_eval = objective_function(new_x)
        if new_eval > current_eval:
            progression.append(new_x)
            current_eval = new_eval
            num_iter_without_improvement = 0
        else:
            num_iter_without_improvement += 1

        if num_iter_without_improvement > max_iter_without_improvement:
            break

        num_iter += 1

    return progression


def probabilistic_hillclimbing(
    *,
    max_iter: int = 1000,
    max_iter_without_improvement: int = 100,
    decay=2,
    rng_seed: int = 1,
):
    rng = np.random.default_rng(rng_seed)
    progression: list[float] = []
    progression.append(rng.normal(0.5, 0.25))
    num_iter = 0
    current_eval = objective_function(progression[-1])
    num_iter_without_improvement = 0

    while num_iter < max_iter:
        new_x = perturb(progression[-1], rng)
        new_eval = objective_function(new_x)
        if rng.uniform(0, 1) < 1 / (1 + np.exp((current_eval - new_eval) / decay)):
            progression.append(new_x)
            current_eval = new_eval
            num_iter_without_improvement = 0
        else:
            num_iter_without_improvement += 1

        if num_iter_without_improvement > max_iter_without_improvement:
            break

        num_iter += 1

    return progression


def simulated_annealing(
    *,
    init_temp: float = 10000,
    min_temp: float = 0.01,
    max_iter: int = 1000,
    max_iter_without_improvement: int = 100,
    temp_decay: float = 0.95,
    rng_seed: int = 1,
) -> list[float]:
    rng = np.random.default_rng(rng_seed)
    progression: list[float] = []
    num_iter = 0
    num_iter_without_improvement = 0
    temp = init_temp

    # Likely in the center of the [0, 1] interval.
    progression.append(rng.normal(0.5, 0.25))
    current_eval = objective_function(progression[-1])

    while num_iter < max_iter:
        x_new = perturb(progression[-1], rng)
        new_eval = objective_function(x_new)
        if new_eval > current_eval or rng.uniform(0, 1) < np.exp(
            (new_eval - current_eval) / temp
        ):
            progression.append(x_new)
            current_eval = new_eval
            num_iter_without_improvement = 0
        else:
            num_iter_without_improvement += 1

        if num_iter_without_improvement > max_iter_without_improvement:
            break

        temp = temp_decay * temp
        if temp < min_temp:
            break

        num_iter += 1

    return progression


def plot_progression(ax, prog: list[float], color: str) -> None:
    t = np.linspace(0, 1, 500)
    ax.plot(t, objective_function(t))
    ax.plot(prog, [objective_function(x) for x in prog], color=color)
    ax.scatter([prog[-1]], [objective_function(prog[-1])], color=color)


def main():
    sim_ann_result = simulated_annealing()
    hc_result = hillclimbing()
    phc_result = probabilistic_hillclimbing()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    plot_progression(ax1, sim_ann_result, "r")
    plot_progression(ax2, hc_result, "g")
    plot_progression(ax3, phc_result, "k")
    plt.show()


if __name__ == "__main__":
    main()
