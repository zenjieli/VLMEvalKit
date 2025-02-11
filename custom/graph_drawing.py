import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_scatter(model_configs: dict[str: dict[str: str|float]],
                 base_model_configs: dict[str: dict[str: str]],
                 scores: dict[str, float],
                 out_img_path: str,
                 caption: str)->None:

    # Create the plot
    plt.figure(figsize=(10, 6))

    for model_name in model_configs:
        if model_name not in scores:
            continue

        if scores[model_name] <= 0:
            continue

        num_params = model_configs[model_name]["num_params"]
        base_model = model_configs[model_name]["base_name"]

        marker = base_model_configs[base_model]["marker"]
        color = base_model_configs[base_model]["color"]
        plt.scatter(num_params, scores[model_name], label=model_name, marker=marker, s=100, c=color)

    # Add legend
    legend_handles = []    
    base_model_set = set(model_configs[model_name]["base_name"] for model_name in scores)
    for base_model in base_model_set:        
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

    # Get the min/max values from scores.values()
    min_score = min(scores.values())
    max_score = max(scores.values())    

    # Find the Y axis range
    upper_bound = (int(max_score*10) + 1) / 10
    lower_bound = (int(min_score*10)) / 10
    plt.xlim(0, 9)
    plt.ylim(lower_bound, upper_bound)

    plt.grid(True)
    plt.savefig(out_img_path)
