from .evaluate import main
from .logic import QuenchThresholds

import optuna


def objective(trial) -> float:
    suggested_thresholds = QuenchThresholds(
        overall_avg=trial.suggest_float("overall_avg", 0.01, 0.5),
        pre_wf_min=trial.suggest_float("pre_wf_min", 0.01, 0.5),
        pre_fwd_min=trial.suggest_float("pre_fwd_min", 0.001, 0.1),
        pre_total_min=trial.suggest_float("pre_total_min", 0.05, 0.5),
        tau_multiplier=trial.suggest_float("tau_multiplier", 0.3, 0.9),
    )

    return main(suggested_thresholds)


if __name__ == "__main__":
    n_trials = 300

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "═" * 50)
    print(" 🏆 OPTIMIZATION COMPLETE 🏆 ".center(50, "═"))
    print("═" * 50)
    print(f"\n🎯 Best Accuracy Achieved: {study.best_value:.2f}%\n")
    print("📊 Optimal Thresholds to use in production:")
    print("-" * 40)
    for key, value in study.best_params.items():
        print(f"   {key:<15} = {value:.5f}")
    print("-" * 40 + "\n")
