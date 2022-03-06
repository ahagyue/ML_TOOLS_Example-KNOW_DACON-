from pathlib import Path
from typing import Dict, Any


class BaseConfig:
    _args: Dict[str, Any]

    def __init__(self):
        self._args = {
            'FOLD': 5,
            'DATA_ROOT': Path('../KNOW_data/'),
            'STRING_INDEX': {
                2017: ['bq4_1a', 'bq4_1b', 'bq4_1c', 'bq5_2', 'bq19_1', 'bq30', 'bq31', 'bq32', 'bq33', 'bq34', 'bq38_1'],
                2018: ['bq4_1a', 'bq4_1b', 'bq4_1c', 'bq5_2', 'bq28_1', 'bq29', 'bq30', 'bq31', 'bq32', 'bq33', 'bq37_1'],
                2019: ['bq4_1a', 'bq4_1b', 'bq4_1c', 'bq5_2', 'bq18_10', 'bq20_1', 'bq22', 'bq23', 'bq24', 'bq27_1'],
                2020: ['bq4_1a', 'bq4_1b', 'bq4_1c', 'bq5_2', 'bq18_10', 'bq20_1']
            },
            'batch_size': 4,
            'num_workers': 1
        }

    def get_config(self, year) -> Dict:
        self._args['STRING_INDEX'] = self._args['STRING_INDEX'][year]
        return self._args