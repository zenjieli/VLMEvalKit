import os.path as osp
import ast
from decord import VideoReader
import numpy as np
import pandas as pd
import PIL.Image as Image
from .video_base import VideoBaseDataset
from ..smp import dump, load, md5, LMUDataRoot


class Virat_MCQ(VideoBaseDataset):

    MD5 = None
    TYPE = 'Video-MCQ'

    def __init__(self, dataset='Virat_MCQ', nframe=8, fps=-1):
        self._root = LMUDataRoot()
        self._video_dir = osp.join(self._root, 'videos/virat')
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['Virat_MCQ']

    def prepare_dataset(self, dataset_name):

        error_msg = None
        tsv_filepath = osp.join(self._root, f'{dataset_name}.tsv')

        if not osp.exists(tsv_filepath):
            error_msg = f'{tsv_filepath} not found'
        elif self.MD5 is not None and self.MD5 != md5(tsv_filepath):
            error_msg = 'MD5 does not match for ' + tsv_filepath
        else:
            data = load(tsv_filepath)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(self._video_dir, item['video'] + '.mp4')):
                    error_msg = f'{item["video"]} not found'
                    break

        if error_msg is not None:
            raise ValueError('Dataset not found or incomplete. ' + error_msg)

        return dict(root=self._root, data_file=tsv_filepath)

    def qa_template(self, data):
        question = 'Question: ' + data['question'] + \
            "Select multiple answers if needed. Answer only with letters, such as 'AB' or 'BD'.\n" + \
            '\n'.join(eval(data['candidates'])) + '\n\nAnswer:'
        return question, data['answer']

    def save_video_frames(self, line):
        vid_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        vid = VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(line['video'])
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(line['video'], len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question, _ = self.qa_template(line)
        message = []
        message.append(dict(type='text', value=question))
        video_path = osp.join(self._video_dir, line['video'] + '.mp4')
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        return message

    @staticmethod
    def map_letter_to_desc(candidates_str: str) -> dict[str, str]:
        """
        Args:
            candidates_str: Like "['A. Closing vehicle trunk', 'B. Exiting vehicle']"
        Returns:
            dict[str, str]: Like {'A': 'Closing vehicle trunk', 'B': 'Exiting vehicle'}
        """
        candidates = ast.literal_eval(candidates_str)
        results = {}
        for option in candidates:
            option_split = [o.strip() for o in option.split('.')]
            results[option_split[0]] = option_split[1]

        return results

    @staticmethod
    def update_tp_fp_fn(results: dict[str, dict[str, int]], gt: str, pred: str, letter_to_desc: dict[str, str]) -> None:
        gt = [c for c in gt if str.isupper(c)]
        pred = [c for c in pred if str.isupper(c)]

        # Ensure all characters are initialized in results
        for c in set(gt + pred):
            desc = letter_to_desc[c]
            if desc not in results:
                results[desc] = {'TP': 0, 'FP': 0, 'FN': 0}

        for c in pred:
            desc = letter_to_desc[c]
            if c in gt:
                results[desc]['TP'] += 1
            else:
                results[desc]['FP'] += 1

        for c in gt:
            if c not in pred:
                desc = letter_to_desc[c]
                results[desc]['FN'] += 1

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)

        results = {}
        for row in data.to_dict(orient='records'):  # Loop over each row as a dict
            letter_to_desc = self.map_letter_to_desc(row['candidates'])
            self.update_tp_fp_fn(results, row['answer'], row['prediction'], letter_to_desc)

        # Compute the precision, recall, and F1 score for each option
        combine_score = {}
        for option, stats in results.items():
            tp = stats['TP']
            fp = stats['FP']
            fn = stats['FN']
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            combine_score[option] = {'f1': f1, 'precision': precision, 'recall': recall}

        # Add a new option 'mean' by averaging each option
        mean_f1 = sum([v['f1'] for v in combine_score.values()]) / len(combine_score)
        mean_precision = sum([v['precision'] for v in combine_score.values()]) / len(combine_score)
        mean_recall = sum([v['recall'] for v in combine_score.values()]) / len(combine_score)
        combine_score['mean'] = {'f1': mean_f1, 'precision': mean_precision, 'recall': mean_recall}

        combine_score = pd.DataFrame.from_dict(combine_score, orient='index')
        score_pth = eval_file.replace('.xlsx', '_score.csv')
        combine_score.to_csv(score_pth, index_label='Activity')

        return combine_score
