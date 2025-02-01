import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_scatter(model_configs: dict[str: dict[str: str|float]],
                 base_model_configs: dict[str: dict[str: str]],
                 scores: list[float],
                 out_img_path: str,
                 caption: str)->None:

    # Create the plot
    plt.figure(figsize=(10, 6))

    for i, model_name in enumerate(model_configs.keys()):
        if scores[i] <= 0:
            continue

        num_params = model_configs[model_name]["num_params"]
        base_model = model_configs[model_name]["base_name"]

        marker = base_model_configs[base_model]["marker"]
        color = base_model_configs[base_model]["color"]
        plt.scatter(num_params, scores[i], label=model_name, marker=marker, s=100, c=color)

    # Add legend
    legend_handles = []
    for base_model in base_model_configs.keys():
        color = base_model_configs[base_model]["color"]
        marker = base_model_configs[base_model]["marker"]
        legend_handle = Line2D([0], [0], marker=marker, color='w', label=base_model,
                               markerfacecolor=color, markersize=10)
        legend_handles.append(legend_handle)

    # Add custom legend
    plt.legend(handles=legend_handles, title="Models", fontsize=12)

    # Add labels and title
    plt.xlabel('Number of trainable parameters (B)', fontsize=14)
    plt.ylabel(caption, fontsize=14)

    plt.xlim(0, 9)
    plt.ylim(0.5, 0.8)

    plt.grid(True)
    plt.savefig(out_img_path)
