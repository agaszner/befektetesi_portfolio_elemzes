import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class PlotResult():

    @classmethod
    def plot_predicted_vs_actual(cls, y_true, y_pred, title='Predicted vs Actual', forecast_horizon=1, start=0, end=200):
        """
        Plot predicted vs actual values.

        :param y_true: Actual values (shape: [samples, forecast_horizon])
        :param y_pred: Predicted values (same shape as y_true)
        :param title: Plot title
        :param forecast_horizon: How many steps ahead you're predicting
        :param start: Index to start plotting from
        :param end: Index to stop plotting at
        """
        sns.set(style="whitegrid", font_scale=1.2)
        plt.figure(figsize=(14, 6))

        if forecast_horizon == 1:
            # Flatten if single-step prediction
            y_true_plot = y_true[start:end].flatten()
            y_pred_plot = y_pred[start:end].flatten()
            x = np.arange(start, end)

            plt.plot(x, y_true_plot, label='Actual', color='dodgerblue', linewidth=2)
            plt.plot(x, y_pred_plot, label='Predicted', color='orange', linestyle='--', linewidth=2)

        else:
            # Plot multiple forecast horizons (overlayed)
            for step in range(forecast_horizon):
                x = np.arange(start, end)
                plt.plot(x, y_true[start:end, step], label=f'Actual t+{step+1}', linestyle='-', alpha=0.6)
                plt.plot(x, y_pred[start:end, step], label=f'Predicted t+{step+1}', linestyle='--', alpha=0.8)

        plt.title(title, fontsize=16)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show()