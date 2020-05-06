import warnings
import os
import glob
import pandas as pd

from nilmtk.api import API
from nilmtk_contrib.disaggregate import Seq2Seq
warnings.filterwarnings("ignore")


def check_models(search_dir: str):

    metrics = []

    files = list(filter(os.path.isfile, glob.glob(search_dir + "seq2seq-temp-weights*")))
    files = sorted(files, key=lambda x: os.path.getmtime(x))

    for model_num in range(0, len(files), 100):
        redd = {
            'power': {
                'mains': ['active'],
                'appliance': ['active']
            },
            'sample_rate': 6,
            'display_predictions': False,
            'artificial_aggregate': False,
            'chunk_size': 200000, # NOTE -> House8 contains 4 chunks with  chunk_size = 2000000
            'appliances': ['unknown'],
            'methods': {
                'Seq2Seq': Seq2Seq({'n_epochs': 30, 'batch_size': 65536, 'chunk_wise_training': True, 'load_model_path': {
                                    'unknown': files[model_num]
                }}),
            },
            'train': {
                'datasets': {
                    'Refit': {
                        'path': './data/data_refit.hdf5',
                        'buildings': { 
                        }
                    }
                }
            },
            'test': {
                'datasets': {
                    'Refit': {
                        'path': './data/data_refit.hdf5',
                        'buildings': {
                            20: {
                                'start_time': '2015-04-22',
                                'end_time': '2015-05-22'
                            },
                        }
                    }
                },
                'metrics': ['mae']
            }
        }

        api_res = API(redd)

        mae_mean = (api_res.errors[0] + api_res.errors[1]) / 2
        metrics.append([files[model_num].split('/')[-1], mae_mean.iloc[0, 0]])

    df_models = pd.DataFrame(metrics, columns=['model', 'score'])
    df_models = df_models.sort_values(by=['score'], ascending=True)
    df_models.to_csv('models_summary.csv')

    print(f'Best model: {df_models.iloc[0, 0]}, score: {df_models.iloc[0, 1]}')


if __name__ == "__main__":
    print('Start')
    # NOTE: Enter your catalog that contains models
    check_models("./")
    print('Done')
