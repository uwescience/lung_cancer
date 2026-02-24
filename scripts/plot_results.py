'''Plotting results from the experiments.'''

from src.bot import Bot

from typing import List

SHOT_DIRS = ["0shot", "4shot", "8shot"]
LEGENDS = ["0-shot", "4-shot", "8-shot"]


def plotROCs():
    """Plot ROC curves comparing 0-shot, 4-shot, and 8-shot experiments."""
    Bot.plotROCs(SHOT_DIRS, legends=LEGENDS)


def plotPredictionRanges(short_dirs: List[str]= SHOT_DIRS, legends: List[str]= LEGENDS):
    """Plot prediction variability (range CDFs) for each experiment set."""
    for shot_dir, legend in zip(short_dirs, legends):
        import matplotlib.pyplot as plt
        Bot.plotPredictionRange(shot_dir, is_plot=False)
        plt.title(f"Prediction Range Distribution — {legend}")
        plt.show()


if __name__ == "__main__":
    plotROCs()
    plotPredictionRanges()
