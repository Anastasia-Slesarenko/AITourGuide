"""
Модуль для визуализации метрик обучения в реальном времени.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os


def _is_notebook() -> bool:
    """Определяет, запущен ли код в Jupyter Notebook / IPython."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        # ZMQInteractiveShell — Jupyter Notebook/Lab/VSCode ipynb
        # TerminalInteractiveShell — обычный IPython в терминале
        return bool(shell_name == "ZMQInteractiveShell")
    except ImportError:
        return False


# Выбираем бэкенд: inline для Jupyter, Agg для скриптов
if _is_notebook():
    # %matplotlib inline уже активирован Jupyter'ом;
    # явно не переключаем бэкенд, чтобы не сломать inline-режим
    pass
else:
    matplotlib.use('Agg')  # Для работы без GUI (скрипты, серверы)


class TrainingPlotter:
    """
    Класс для создания и обновления графиков метрик обучения.
    """

    # Константы класса
    SMOOTHING_WINDOW: int = 10
    SAVE_DPI: int = 150
    OVERVIEW_FIGSIZE: tuple = (16, 10)
    COMPARISON_FIGSIZE: tuple = (12, 6)

    def __init__(self, output_dir: str, experiment_name: str):
        """
        Args:
            output_dir: Директория для сохранения графиков
            experiment_name: Название эксперимента
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # Кэшируем результат один раз,
        # чтобы не вызывать get_ipython() при каждом рендере
        self._in_notebook: bool = _is_notebook()

        # Настройка стиля
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'train_loss': '#1f77b4',
            'eval_loss': '#ff7f0e',
            'eval_hit@1': '#2ca02c',
            'eval_mrr': '#d62728',
            'eval_none_accuracy': '#9467bd',
            'eval_hard_accuracy': '#8c564b',
            'eval_easy_accuracy': '#e377c2',
            'eval_semi_hard_accuracy': '#7f7f7f',
        }

    # Хелпер: рисует одну линию метрики с защитой от несовпадения длин
    def _plot_metric(
        self,
        ax,
        steps: List[int],
        metrics_history: Dict[str, List[float]],
        key: str,
        label: str,
        marker: str,
        linewidth: float = 2,
        markersize: int = 4,
        color: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Рисует линию метрики key на оси ax.

        Возвращает True, если линия была нарисована.
        Защищает от несовпадения длин steps и значений метрики.
        """
        if key not in metrics_history or not metrics_history[key]:
            return False
        values = metrics_history[key]
        aligned_steps = steps[: len(values)]
        plot_kwargs: dict = dict(
            label=label,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            **kwargs,
        )
        if color is not None:
            plot_kwargs['color'] = color
        ax.plot(aligned_steps, values, **plot_kwargs)
        return True


    def plot_metrics(
        self,
        steps: List[int],
        metrics_history: Dict[str, List[float]],
        train_loss_steps: Optional[List[int]] = None,
        train_loss_values: Optional[List[float]] = None,
    ):
        """
        Создает и сохраняет графики всех метрик.

        Args:
            steps: Шаги для eval метрик
            metrics_history: История eval метрик
            train_loss_steps: Шаги для train loss (опционально)
            train_loss_values: Значения train loss (опционально)
        """
        # Очищаем предыдущий вывод один раз перед обоими графиками
        if self._in_notebook:
            from IPython.display import display, clear_output
            clear_output(wait=True)

        # Создаем фигуру с несколькими subplot'ами
        fig = plt.figure(figsize=self.OVERVIEW_FIGSIZE)

        # 1. Train Loss + Eval Loss на одном графике
        ax1 = plt.subplot(2, 3, 1)
        all_loss_vals: List[float] = []
        if train_loss_steps and train_loss_values:
            train_loss_steps_arr = np.array(train_loss_steps)
            # Сглаживание: rolling с min_periods=1 сохраняет все точки,
            # включая начало кривой (в отличие от mode='valid')
            window = min(self.SMOOTHING_WINDOW, len(train_loss_values))
            smoothed = (
                pd.Series(train_loss_values)
                .rolling(window, min_periods=1)
                .mean()
                .to_numpy()
            )
            ax1.plot(
                train_loss_steps_arr,
                train_loss_values,
                label='Train Loss (raw)',
                color=self.colors['train_loss'],
                linewidth=1,
                alpha=0.3,
            )
            ax1.plot(
                train_loss_steps_arr,
                smoothed,
                label='Train Loss (smooth)',
                color=self.colors['train_loss'],
                linewidth=2,
                alpha=0.9,
            )
            all_loss_vals.extend(train_loss_values)
        eval_loss_vals = metrics_history.get('eval_loss', [])
        if len(eval_loss_vals) > 0 and steps:
            ax1.plot(
                steps[: len(eval_loss_vals)],
                eval_loss_vals,
                label='Eval Loss',
                color=self.colors['eval_loss'],
                linewidth=2,
                marker='o',
                markersize=4,
                alpha=0.9,
            )
            all_loss_vals.extend(eval_loss_vals)
        ax1.set_xlabel('Training Steps', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.set_title('Train vs Eval Loss', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        # Масштаб по 95-му перцентилю: выбросы в начале обучения
        # не сжимают основную часть кривой
        if all_loss_vals:
            top = float(np.percentile(all_loss_vals, 95)) * 1.2 + 0.1
            ax1.set_ylim(bottom=0, top=top)

        # 2. Hit@1 и MRR
        ax2 = plt.subplot(2, 3, 2)
        # FIX: ключ в metrics_history — 'eval_hit_1' (из train.py),
        # а не 'eval_hit@1' (@ недопустим в именах Python-атрибутов)
        self._plot_metric(
            ax2, steps, metrics_history, 'eval_hit_1', 'Hit@1', 'o',
            color=self.colors['eval_hit@1'],
        )
        self._plot_metric(
            ax2, steps, metrics_history, 'eval_mrr', 'MRR', 's',
            color=self.colors['eval_mrr'],
        )
        # Маркер лучшего чекпоинта по Hit@1
        hit1_vals = metrics_history.get('eval_hit_1', [])
        if hit1_vals and steps:
            best_idx = int(np.argmax(hit1_vals))
            best_step = steps[best_idx]
            best_val = hit1_vals[best_idx]
            ax2.axvline(
                best_step, color='gray', linestyle='--',
                linewidth=1, alpha=0.6,
            )
            ax2.annotate(
                f'best Hit@1={best_val:.3f}\n@ step {best_step}',
                xy=(best_step, best_val),
                xytext=(8, -20),
                textcoords='offset points',
                fontsize=8,
                color='gray',
            )
        ax2.set_xlabel('Training Steps', fontsize=10)
        ax2.set_ylabel('Score', fontsize=10)
        ax2.set_title('Ranking Metrics', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim((0, 1.05))

        # 3. None Accuracy
        ax3 = plt.subplot(2, 3, 3)
        self._plot_metric(
            ax3, steps, metrics_history, 'eval_none_accuracy',
            'None Accuracy', 'D',
            color=self.colors['eval_none_accuracy'],
        )
        ax3.set_xlabel('Training Steps', fontsize=10)
        ax3.set_ylabel('Accuracy', fontsize=10)
        ax3.set_title(
            'None-of-the-Above Accuracy', fontsize=12, fontweight='bold',
        )
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim((0, 1.05))

        # 4. Hard Accuracy
        ax4 = plt.subplot(2, 3, 4)
        self._plot_metric(
            ax4, steps, metrics_history, 'eval_hard_accuracy',
            'Hard Accuracy', '^',
            color=self.colors['eval_hard_accuracy'],
        )
        ax4.set_xlabel('Training Steps', fontsize=10)
        ax4.set_ylabel('Accuracy', fontsize=10)
        ax4.set_title(
            'Hard Examples Accuracy', fontsize=12, fontweight='bold',
        )
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim((0, 1.05))

        # 5. Easy Accuracy
        ax5 = plt.subplot(2, 3, 5)
        self._plot_metric(
            ax5, steps, metrics_history, 'eval_easy_accuracy',
            'Easy Accuracy', 'v',
            color=self.colors['eval_easy_accuracy'],
        )
        ax5.set_xlabel('Training Steps', fontsize=10)
        ax5.set_ylabel('Accuracy', fontsize=10)
        ax5.set_title(
            'Easy Examples Accuracy', fontsize=12, fontweight='bold',
        )
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim((0, 1.05))

        # 6. Semi-Hard Accuracy
        ax6 = plt.subplot(2, 3, 6)
        self._plot_metric(
            ax6, steps, metrics_history, 'eval_semi_hard_accuracy',
            'Semi-Hard Accuracy', '<',
            color=self.colors['eval_semi_hard_accuracy'],
        )
        ax6.set_xlabel('Training Steps', fontsize=10)
        ax6.set_ylabel('Accuracy', fontsize=10)
        ax6.set_title(
            'Semi-Hard Examples Accuracy', fontsize=12, fontweight='bold',
        )
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim((0, 1.05))

        # Общий заголовок
        fig.suptitle(
            f'Training Metrics: {self.experiment_name}',
            fontsize=14,
            fontweight='bold',
            y=0.995,
        )

        plt.tight_layout(rect=(0, 0, 1, 0.99))

        # Сохраняем
        plot_path = os.path.join(self.plots_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=self.SAVE_DPI, bbox_inches='tight')

        # Отображаем inline в Jupyter Notebook
        if self._in_notebook:
            display(fig)

        plt.close(fig)

        print(f"График сохранен: {plot_path}")

    def plot_comparison(
        self,
        experiments: Dict[str, Dict[str, List]],
        metric_name: str = 'eval_hit@1',
    ):
        """
        Сравнивает несколько экспериментов на одном графике.

        Args:
            experiments: {exp_name: {'steps': [...], 'values': [...]}}
            metric_name: Название метрики для сравнения
        """
        fig, ax = plt.subplots(figsize=self.COMPARISON_FIGSIZE)

        for exp_name, data in experiments.items():
            ax.plot(
                data['steps'],
                data['values'],
                label=exp_name,
                linewidth=2,
                marker='o',
                markersize=4,
            )

        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(
            f'Experiments Comparison: {metric_name}',
            fontsize=14,
            fontweight='bold',
        )
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(
            self.plots_dir, f'comparison_{metric_name}.png',
        )
        plt.savefig(plot_path, dpi=self.SAVE_DPI, bbox_inches='tight')

        # Отображаем inline в Jupyter Notebook
        if self._in_notebook:
            from IPython.display import display
            display(fig)

        plt.close(fig)

        print(f"График сравнения сохранен: {plot_path}")
