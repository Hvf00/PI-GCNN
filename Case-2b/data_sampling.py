#Data_Sampling
import numpy as np
import matplotlib.pyplot as plt

def sine_wave_pattern_sampling(x_data, y_data, xy, u, v, n_samples=10000, amplitude=0.4, frequency=3, plot=True, save_path="Results/plots/selected_points.png"):
    L = x_data.max() - x_data.min()

    # Pattern sample points
    x_pattern = np.linspace(x_data.min(), x_data.max(), n_samples)
    y_pattern = amplitude * np.sin(2 * np.pi * frequency * x_pattern / L)

    # Smooth curve for plotting
    x_curve = np.linspace(x_data.min(), x_data.max(), 500)
    y_curve = amplitude * np.sin(2 * np.pi * frequency * x_curve / L)

    # Find nearest real data points
    indices = []
    for xp, yp in zip(x_pattern, y_pattern):
        distances = np.sqrt((x_data - xp)**2 + (y_data - yp)**2)
        indices.append(np.argmin(distances))
    indices = np.array(indices)

    xy_sampled = xy[indices]
    u_sampled = u[indices]
    v_sampled = v[indices]

    if plot:
        plt.figure(figsize=(12, 2))
        plt.plot(x_curve, y_curve, linewidth=1, label="Sine Pattern Curve")
        plt.scatter(xy_sampled[:, 0], xy_sampled[:, 1], s=8, color="red", label="Selected Points")
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.title("Sine Wave Pattern Sampling")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.savefig(save_path, dpi=300)
        plt.show()

    return indices, xy_sampled, u_sampled, v_sampled, (x_pattern, y_pattern), (x_curve, y_curve)

def tanh_pattern_sampling(x_data, y_data, xy, u, v, n_samples=40, amplitude=0.4, steepness=1.0, shift=5.0, plot=True, save_path="selected_points.png"):
    # Pattern sample points
    x_pattern = np.linspace(x_data.min(), x_data.max(), n_samples)
    y_pattern = amplitude * np.tanh(steepness * (x_pattern - shift))

    # Smooth curve for plotting
    x_curve = np.linspace(x_data.min(), x_data.max(), 500)
    y_curve = amplitude * np.tanh(steepness * (x_curve - shift))

    # Find nearest real data points
    indices = []
    for xp, yp in zip(x_pattern, y_pattern):
        distances = np.sqrt((x_data - xp)**2 + (y_data - yp)**2)
        indices.append(np.argmin(distances))
    indices = np.array(indices)

    xy_sampled = xy[indices]
    u_sampled = u[indices]
    v_sampled = v[indices]

    if plot:
        plt.figure(figsize=(12, 2))
        plt.plot(x_curve, y_curve, linewidth=1, label="Tanh Pattern Curve")
        plt.scatter(xy_sampled[:, 0], xy_sampled[:, 1], s=8, color="red", label="Selected Points")
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.title("Tanh Pattern Sampling")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.savefig(save_path, dpi=300)
        plt.show()

    return indices, xy_sampled, u_sampled, v_sampled, (x_pattern, y_pattern), (x_curve, y_curve)



def random_fixed_sampling(xy, u, v, n_samples=1000, seed=42, plot=True, save_path="selected_points.png"):
    np.random.seed(seed)
    indices = np.random.choice(len(xy), size=n_samples, replace=False)
    
    xy_sampled = xy[indices]
    u_sampled = u[indices]
    v_sampled = v[indices]
    
    if plot:
        plt.figure(figsize=(12, 2))
        plt.scatter(xy_sampled[:, 0], xy_sampled[:, 1], s=8, color="red", label="Random Fixed Points")
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.title("Fixed Random Sampling")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.savefig(save_path, dpi=300)
        plt.show()
    
    return indices, xy_sampled, u_sampled, v_sampled
