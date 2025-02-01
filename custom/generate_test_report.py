import os
import os.path as osp
import csv
import re

from custom.graph_drawing import plot_scatter


def extract_score_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line
        second_line = next(reader)  # Read the second line
        for item in second_line:
            if re.match(r'^-?\d+(\.\d+)?$', item):  # Check if the item is a number
                score = float(item)
                return score
    return None


def extract_test_scores(base_dir: str, model_name: str) -> dict[str: float]:
    model_dir = osp.join(base_dir, model_name)
    if not osp.exists(model_dir):
        return {}

    test_scores = {}
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.csv'):
            test_name = file_name[len(model_name) + 1:].split('_')[0]
            file_path = osp.join(model_dir, file_name)
            score = extract_score_from_csv(file_path)
            if score is not None:
                if score > 100:  # Range [0,2000]
                    score /= 2000
                elif score > 1:  # Range [0,100]
                    score /= 100
                test_scores[test_name] = score
    return test_scores


def find_highest_score_models(model_names: list[str], test_types: list[str], test_scores: dict[str: dict[str: float]]):
    """
    Finds the model with the highest score for each test type.
    """
    highest_score_models = {}
    for test_type in test_types:
        highest_score_models[test_type] = max(
            (model_name for model_name in model_names if test_type in test_scores[model_name]),
            key=lambda model_name: test_scores[model_name][test_type],
            default=None
        )
    return highest_score_models


def generate_markdown_table(model_names: list[str],
                            test_scores: dict[dict[str: float]],
                            test_types: list[str],
                            highest_score_models: dict[str: str]):
    markdown_table = "| Model Name | " + " | ".join(test_types) + " |\n"
    markdown_table += "| " + " --- | " * (len(test_types) + 1) + "\n"

    for model_name in model_names:
        row = [model_name]
        for test_type in test_types:
            score_str = f"{test_scores[model_name].get(test_type, 0):.2f}"
            if model_name == highest_score_models[test_type]:
                score_str = "**" + score_str + "**"

            row.append(score_str)
        markdown_table += "| " + " | ".join(row) + " |\n"

    return markdown_table


def generate_test_report(base_dir: str,
                         model_names: list[str],
                         custom_test_types: list[str],
                         output_file: str) -> dict[str: dict[str: float]]:
    all_test_types = set()
    test_scores = {}

    for model_name in model_names:
        test_scores[model_name] = extract_test_scores(base_dir, model_name)
        all_test_types.update(test_scores[model_name].keys())

    standard_test_types = [test_type for test_type in all_test_types if test_type not in custom_test_types]
    standard_test_types = sorted(standard_test_types)

    # Add a average test score for each model. Exclude custom test types in the average.
    all_test_types.add("Average")
    for model_name in model_names:
        average_score = sum(test_scores[model_name].get(test_type, 0)
                            for test_type in standard_test_types) / len(standard_test_types)
        test_scores[model_name]["Average"] = average_score

    highest_score_models = find_highest_score_models(model_names, all_test_types, test_scores)

    report_text = generate_markdown_table(
        model_names, test_scores, standard_test_types + ["Average"], highest_score_models)
    report_text += "\n\n" + generate_markdown_table(model_names, test_scores, custom_test_types, highest_score_models)
    with open(output_file, 'w') as f:
        f.write(report_text)

    return test_scores


def append_graphs(model_configs: dict[str:dict[str:str | float]],
                  base_model_configs: dict[str:dict[str:str]],
                  test_scores: dict[str:dict[str:float]],
                  report_filepath: str) -> None:
    model_names = list(model_configs.keys())

    # Average scores
    scores = [test_scores[model_name]["Average"] for model_name in model_names]
    score_img_filename = "score_vs_params.png"
    plot_scatter(model_configs, base_model_configs, scores,
                 osp.join(osp.dirname(report_filepath), score_img_filename),
                 "Mean score vs parameters")

    # HICO scores
    scores = [test_scores[model_name].get("HICO", 0) for model_name in model_names]
    hico_score_img_filename = "hico_score_vs_params.png"
    plot_scatter(model_configs, base_model_configs, scores,
                 osp.join(osp.dirname(report_filepath), hico_score_img_filename),
                 "HICO scoare vs parameters")

    # Append the graph to the report file
    with open(report_filepath, "r") as f:
        markdown_text = f.read()
    markdown_text += f"\n\n![Mean score vs parameters]({score_img_filename})\n"
    markdown_text += f"\n\n![HICO score vs parameters]({hico_score_img_filename})\n"
    with open(report_filepath, "w") as f:
        f.write(markdown_text)


def get_model_configs():
    model_configs = {
        "InternVL2_5-1B": {"base_name": "InternVL2_5", "num_params": 0.9},
        "InternVL2_5-2B": {"base_name": "InternVL2_5", "num_params": 2.1},
        "InternVL2_5-4B": {"base_name": "InternVL2_5", "num_params": 3.5},
        "InternVL2_5-8B": {"base_name": "InternVL2_5", "num_params": 7.5},
        "MiniCPM-V-2_6": {"base_name": "MiniCPM-V-2_6", "num_params": 7.9},
        "Llama-3-VILA1.5-8b": {"base_name": "Llama-3-VILA1.5-8b", "num_params": 8.4},
        "llava_next_llama3": {"base_name": "llava_next_llama3", "num_params": 1.5},
        "Phi-3.5-Vision": {"base_name": "Phi-3.5-Vision", "num_params": 3.9},
        "Qwen2-VL-2B-Instruct": {"base_name": "Qwen2-VL", "num_params": 2.1},
        "Qwen2-VL-7B-Instruct": {"base_name": "Qwen2-VL", "num_params": 7.7},
        "Qwen2.5-VL-3B": {"base_name": "Qwen2.5-VL", "num_params": 3.5},
        "Qwen2.5-VL-7B": {"base_name": "Qwen2.5-VL", "num_params": 7.7},
        "VideoLLaMA3-2B-Image": {"base_name": "VideoLLaMA3", "num_params": 1.8},
        "VideoLLaMA3-7B-Image": {"base_name": "VideoLLaMA3", "num_params": 7.5}}

    base_model_configs = {"InternVL2_5": {"marker": "o", "color": "blue"},
                          "MiniCPM-V-2_6": {"marker": "s", "color": "green"},
                          "Llama-3-VILA1.5-8b": {"marker": "^", "color": "red"},
                          "llava_next_llama3": {"marker": "v", "color": "purple"},
                          "Phi-3.5-Vision": {"marker": ">", "color": "orange"},
                          "Qwen2-VL": {"marker": "<", "color": "cyan"},
                          "Qwen2.5-VL": {"marker": "p", "color": "magenta"},
                          "VideoLLaMA3": {"marker": "h", "color": "brown"}}
    return model_configs, base_model_configs


if __name__ == "__main__":
    custom_test_types = ["HICO"]
    report_filepath = "custom/report/test_report.md"
    model_configs, base_model_configs = get_model_configs()
    test_scores = generate_test_report("outputs", model_configs.keys(), custom_test_types, report_filepath)
    append_graphs(model_configs, base_model_configs, test_scores, report_filepath)
