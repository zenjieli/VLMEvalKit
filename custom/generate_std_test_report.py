"""Generate standard benchmark test report
"""

import os
import os.path as osp
import csv
import re

from graph_drawing import plot_scatter


def get_model_configs():
    model_configs = {
        'InternVL2_5-1B': {'base_name': 'InternVL2_5', 'num_params': 0.9},
        'InternVL2_5-2B': {'base_name': 'InternVL2_5', 'num_params': 2.1},
        'InternVL2_5-4B': {'base_name': 'InternVL2_5', 'num_params': 3.5},
        'InternVL2_5-8B': {'base_name': 'InternVL2_5', 'num_params': 7.5},
        'MiniCPM-V-2_6': {'base_name': 'MiniCPM-V-2_6', 'num_params': 7.9},
        'Llama-3-VILA1.5-8b': {'base_name': 'Llama-3-VILA1.5-8b', 'num_params': 8.4},
        'llava_next_llama3': {'base_name': 'llava_next_llama3', 'num_params': 1.5},
        'Phi-3.5-Vision': {'base_name': 'Phi-3.5-Vision', 'num_params': 3.9},
        'Qwen2-VL-2B-Instruct': {'base_name': 'Qwen2-VL', 'num_params': 2.1},
        'Qwen2-VL-7B-Instruct': {'base_name': 'Qwen2-VL', 'num_params': 7.7},
        'Qwen2.5-VL-3B': {'base_name': 'Qwen2.5-VL', 'num_params': 3.5},
        'Qwen2.5-VL-7B': {'base_name': 'Qwen2.5-VL', 'num_params': 7.7},
        'Janus-Pro-7B': {'base_name': 'Janus-Pro', 'num_params': 7.5},
        'VideoLLaMA3-2B-Image': {'base_name': 'VideoLLaMA3', 'num_params': 1.8},
        'VideoLLaMA3-7B-Image': {'base_name': 'VideoLLaMA3', 'num_params': 7.5},
        'VideoLLaMA3-2B': {'base_name': 'VideoLLaMA3', 'num_params': 2},
        'VideoLLaMA3-7B': {'base_name': 'VideoLLaMA3', 'num_params': 7.5}}

    base_model_configs = {'InternVL2_5': {'marker': 'o', 'color': 'blue'},
                          'MiniCPM-V-2_6': {'marker': 's', 'color': 'green'},
                          'Llama-3-VILA1.5-8b': {'marker': '^', 'color': 'red'},
                          'llava_next_llama3': {'marker': 'v', 'color': 'purple'},
                          'Phi-3.5-Vision': {'marker': '>', 'color': 'orange'},
                          'Qwen2-VL': {'marker': '<', 'color': 'cyan'},
                          'Qwen2.5-VL': {'marker': 'p', 'color': 'magenta'},
                          'Janus-Pro': {'marker': '*', 'color': 'pink'},
                          'VideoLLaMA3': {'marker': 'h', 'color': 'brown'}}
    return model_configs, base_model_configs


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


def extract_test_scores(base_dir: str, model_name: str, test_names_lower: list[str]) -> dict[str: float]:
    model_dir = osp.join(base_dir, model_name)
    if not osp.exists(model_dir):
        return {}

    test_scores = {}
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.csv'):  # Like InternVL2_5-1B_MMBench_DEV_EN_acc.csv
            items = file_name[len(model_name) + 1:].split('_')
            test_name = "_".join(items[0:-1])
            if test_name.lower() not in test_names_lower:
                continue

            file_path = osp.join(model_dir, file_name)
            score = extract_score_from_csv(file_path)
            if score is not None:
                if score > 100:  # Range [0,2000], MME
                    score /= 2000
                elif score > 1:  # Range [0,100]
                    score /= 100
                test_scores[test_name] = score
    return test_scores


def find_highest_score_models(test_types: list[str], test_scores: dict[str: dict[str: float]]):
    """
    Finds the model with the highest score for each test type.
    """
    highest_score_models = {}
    for test_type in test_types:
        highest_score_models[test_type] = max(
            (model_name for model_name in test_scores if test_type in test_scores[model_name]),
            key=lambda model_name: test_scores[model_name][test_type],
            default=None)
    return highest_score_models


def generate_markdown_table(model_names: list[str],
                            test_scores: dict[dict[str: float]],
                            test_types: list[str],
                            highest_score_models: dict[str: str]):
    markdown_table = '| Model Name | ' + ' | '.join(test_types) + ' |\n'
    markdown_table += '| ' + ' --- | ' * (len(test_types) + 1) + '\n'

    for model_name in test_scores:
        row = [model_name]
        for test_type in test_types:
            score_str = f'{test_scores[model_name].get(test_type, 0):.2f}'
            if model_name == highest_score_models[test_type]:
                score_str = '**' + score_str + '**'

            row.append(score_str)
        markdown_table += '| ' + ' | '.join(row) + ' |\n'

    return markdown_table


def generate_test_report(base_dir: str,
                         model_names: list[str],
                         test_names: list[str],
                         output_file: str,
                         overwrite_existing: bool = False) -> dict[str: dict[str: float]]:
    all_test_types = set()
    test_scores = {}

    test_names_lower = [test_name.lower() for test_name in test_names]
    for model_name in model_names:
        model_test_scores = extract_test_scores(base_dir, model_name, test_names_lower)
        if len(model_test_scores) == 0:
            continue

        test_scores[model_name] = model_test_scores
        all_test_types.update(test_scores[model_name].keys())

    standard_test_types = list(all_test_types)
    standard_test_types = sorted(standard_test_types)

    # Add a average test score for each model
    if len(standard_test_types) > 1:
        standard_test_types += ['Mean']
        for model_name in model_names:
            Mean_score = sum(test_scores[model_name].get(test_type, 0)
                             for test_type in standard_test_types) / len(standard_test_types)
            test_scores[model_name]['Mean'] = Mean_score

    highest_score_models = find_highest_score_models(standard_test_types, test_scores)

    report_text = generate_markdown_table(
        model_names, test_scores, standard_test_types, highest_score_models)
    with open(output_file, 'w' if overwrite_existing else 'a') as f:
        f.write(report_text)

    return test_scores


def append_graphs(model_configs: dict[str:dict[str:str | float]],
                  base_model_configs: dict[str:dict[str:str]],
                  test_scores: dict[str:dict[str:float]],
                  report_filepath: str) -> None:

    # Get scores
    scores = {}  # model_name: score
    score_name = ''
    for model_name, test_score in test_scores.items():
        if 'Mean' in test_score:
            if score_name and score_name != 'Mean':
                raise ValueError(f'Inconsistent score names: {test_score}')
            score_name = 'Mean'
        elif len(test_score) == 1:
            score_name = next(iter(test_score.keys()))
        else:
            raise ValueError(f'Invalid test scores: {test_score}')

        scores[model_name] = test_score[score_name]

    score_img_filename = f'{score_name}_score_vs_params.png'
    plot_scatter(model_configs, base_model_configs, scores,
                 osp.join(osp.dirname(report_filepath), score_img_filename),
                 f'{score_name} score vs parameters')

    # Append the graph to the report file
    markdown_text = f'\n\n![{score_name} score vs parameters]({score_img_filename})\n'
    with open(report_filepath, 'a') as f:
        f.write(markdown_text)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_models', type=str, nargs='+', required=True)
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--report_name', type=str, default='test_report')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    report_filepath = f'custom/report/{args.report_name}.md'
    model_configs, base_model_configs = get_model_configs()

    base_models = args.base_models
    model_names = [model_name for model_name, config in model_configs.items() if config['base_name'] in base_models]

    test_scores = generate_test_report('outputs', model_names, test_names=args.data, output_file=report_filepath)
    append_graphs(model_configs, base_model_configs, test_scores, report_filepath)
