import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
from arch import arch_model
import os
import time


portfolio=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Portfolio_agg.csv')
window_size = 504  # 3 weeks
shift = 24  # shift by 24 hours

folder_path = r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Agg_data'

for filename in os.listdir(folder_path):
    data=pd.read_csv(os.path.join(folder_path,filename))

    Egarch=arch_model(window_data, vol='EGARCH', p=1, q=1),