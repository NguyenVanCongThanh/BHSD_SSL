import os
from kaggle.api.kaggle_api_extended import KaggleApi

# 1. Test when KAGGLE_CONFIG_DIR is NOT set (Account 1 default)
if 'KAGGLE_CONFIG_DIR' in os.environ:
    del os.environ['KAGGLE_CONFIG_DIR']
api1 = KaggleApi()
api1.authenticate()
print("[Acc 1 Check] Authenticated as:", api1.get_config_value(api1.CONFIG_NAME_USER))

# 2. Test when KAGGLE_CONFIG_DIR is set to .kaggle_acc2 (Account 2)
os.environ['KAGGLE_CONFIG_DIR'] = '/home/u001015/BHSD_SSL/kaggle_mean_teacher/.kaggle_acc2'
api2 = KaggleApi()
api2.authenticate()
print("[Acc 2 Check] Authenticated as:", api2.get_config_value(api2.CONFIG_NAME_USER))
