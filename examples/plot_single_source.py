"""plot relatice dimension from single source"""
import matplotlib.pyplot as plt

from dyngdim.plotting import plot_single_source
from dyngdim.io import load_single_source_results

results = load_single_source_results()
plot_single_source(results)

plt.show()
