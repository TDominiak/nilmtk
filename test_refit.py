import warnings
warnings.filterwarnings("ignore")
from nilmtk.api import API
from nilmtk.disaggregate import Mean
from nilmtk_contrib.disaggregate import DAE,Seq2Point, Seq2Seq, RNN, WindowGRU

redd = {
    'power': {
        'mains': ['active'],
        'appliance': ['active']
    },
    'sample_rate': 6,
    'display_predictions': True,
    'artificial_aggregate': False,
    'chunk_size': 2000000, # NOTE -> House8 contains 4 chunks with  chunk_size = 2000000
    'appliances': ['unknown'],
    'methods': {
        'Seq2Seq': Seq2Seq({'n_epochs': 1, 'batch_size': 32, 'chunk_wise_training': True, 'load_model_path': {
                             'unknown': '/home/tomasz/lerta/consumption/nilmtk/seq2seq-temp-weights-36375.h5' # NOTE: In building 20 kettle -> unknown
        }}),
    },
    'train': {
        'datasets': {
            'Refit': {
                'path': '/home/tomasz/lerta/consumption/nilmtk/data/data_refit.hdf5',
                'buildings': {
                    2: {
                        'start_time': '2013-09-18',
                        'end_time': '2015-01-05'
                    },
                    3: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-05'
                    },
                    4: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-05'
                    },
                    5: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-05'
                    },
                    6: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-05'
                    },
                    7: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-05'
                    },
                    8: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-05'
                    },
                    9: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-05'
                    },
                    11: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-05'
                    },
                    12: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-01-05'
                    },
                }
            }
        }
    },
    'test': {
        'datasets': {
            'Refit': {
                'path': '/home/tomasz/lerta/consumption/nilmtk/data/data_refit.hdf5',
                'buildings': {
                    20: {
                        'start_time': '2015-05-22',
                        'end_time': '2015-06-22'
                    },
                }
            }
        },
        'metrics': ['mae']
    }
}

api_res = API(redd)

api_res.errors

api_res.errors_keys