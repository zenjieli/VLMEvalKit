import pandas as pd
from .image_base import ImageBaseDataset
from ..smp import load, dump

class HICOVQADataset(ImageBaseDataset):
    TYPE = 'MCQ'
    DATASET_URL = {
        'HICO': 'https://huggingface.co/datasets/hoveringgull/hico_tsv/resolve/main/hico.tsv',
        'HICO_Mini': 'https://huggingface.co/datasets/hoveringgull/hico_tsv/resolve/main/hico_mini.tsv'
    }
    DATASET_MD5 = {
        'HICO': '0b672999f634f5d5155b9a86907fa4e6',
        'HICO_Mini': '7c176883f0c282a5ae438ad9d300f1c2'
    }

    @staticmethod
    def F1_score(tp, fp, fn):
        return tp/(tp + 0.5*(fp+fn))

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)

        tp = fp = fn = 0
        for row in data.to_dict(orient='records'): # Loop over each row as a dict
            ground_truth = {c for c in row['answer']}
            model_output = {c for c in row['prediction'] if c.isupper()}

            tp += len(ground_truth.intersection(model_output))
            fp += len(model_output.difference(ground_truth))
            fn += len(ground_truth.difference(model_output))

        combine_score = {'f1': self.F1_score(tp, fp, fn)}
        combine_score = pd.DataFrame([combine_score])
        score_pth = eval_file.replace('.xlsx', '_score.csv')
        dump(combine_score, score_pth)

        return combine_score