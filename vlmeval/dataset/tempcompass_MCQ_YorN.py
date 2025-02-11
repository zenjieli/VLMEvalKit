from .video_concat_dataset import ConcatVideoDataset
from .utils.tempcompass import *

class TempCompassMCQ_YorN(ConcatVideoDataset):
    def __init__(self, dataset='TempCompass_MCQ_YorN', nframe=0, fps=-1):        
        self.DATASET_SETS[dataset] = ['TempCompass_MCQ', 'TempCompass_YorN']
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['TempCompass']

    def evaluate(self, eval_file, **judge_kwargs):
        result = super().evaluate(eval_file=eval_file, **judge_kwargs)
        suffix = eval_file.split('.')[-1]
        result = result.reset_index().rename(columns={'index': 'dim.task_type'})
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        avg_dict = {}
        for idx, item in result.iterrows():
            dim, task_type = item['dim.task_type'].split('. ')
            if dim not in avg_dict:
                avg_dict[dim] = {'success': 0.0, 'overall': 0.0}
            if task_type not in avg_dict:
                avg_dict[task_type] = {'success': 0.0, 'overall': 0.0}
            if 'overall' not in avg_dict:
                avg_dict['overall'] = {'success': 0.0, 'overall': 0.0}
            avg_dict[dim]['success'] += item['success']
            avg_dict[dim]['overall'] += item['overall']
            avg_dict[task_type]['success'] += item['success']
            avg_dict[task_type]['overall'] += item['overall']
            avg_dict['overall']['success'] += item['success']
            avg_dict['overall']['overall'] += item['overall']
            result.loc[idx, 'acc'] = round(item['success'] / item['overall'] * 100, 2)
        for key, value in avg_dict.items():
            # 使用 loc 方法添加新行
            result.loc[len(result)] = {
                'dim.task_type': key,
                'success': value['success'],
                'overall': value['overall'],
                'acc': round(value['success'] / value['overall'] * 100, 2)
            }
        dump(result, score_file)
        return result