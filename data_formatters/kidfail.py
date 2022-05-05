# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Volatility dataset.

Defines dataset specific column definitions and data transformations.
"""
# %%


import sys
import os
pathProject = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#pathProject = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pathProject

try:
    os.chdir(pathProject)
except:
    os.chdir('/app/')
    pathProject = '/app/'

sys.path.append('src')



from tft_tf2.data_formatters import base
import tft_tf2.libs.utils as utils
import sklearn.preprocessing
import pandas as pd
import numpy as np
import sklearn.preprocessing

GenericDataFormatter = base.GenericDataFormatter
DataTypes = base.DataTypes
InputTypes = base.InputTypes
# %%

class KidfailFormatter(GenericDataFormatter):
  """Defines and formats data for the volatility dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
    ('id_combined', DataTypes.CATEGORICAL, InputTypes.ID),
    ('x_date', DataTypes.DATE, InputTypes.TIME),
    ('lab_ALAT_ALT_GPT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_ASAT_AST_GOT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Albumin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Ammoniak', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_BNP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_C_reaktives_Protein', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Ca_ion', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Calcium_gesamt', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Chlorid_im_Blut', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Eisen', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Erythrozyten', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Fibrinogen', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Folsaeure', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_GFR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_GGT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Glukose', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Haematokrit', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Haemoglobin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_HbA1c', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Kalium', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Kreatinin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Lactat', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Leukozyten', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_MCH', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_MCHC', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_MCV', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Natrium', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_PTT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Quick', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_RDW', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Thrombozyten', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Vitamin_B12', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_pCO2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_pH_im_Blut', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_pO2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_sO2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('lab_Interleukin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ('ops_autoenc_0', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_6', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_7', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_8', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_9', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_11', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_12', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('ops_autoenc_13', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('fab_autoenc_0', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('fab_autoenc_1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('fab_autoenc_2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('fab_autoenc_3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('fab_autoenc_4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('fab_autoenc_5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('fab_autoenc_6', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT), 
    ('fall_age', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
    ('fall_female', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    #('fall_kreaNormRangeMin', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
    #('fall_kreaNormRangeMax', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
    ('icd', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ('icd_17_9x', DataTypes.CATEGORICAL, InputTypes.TARGET),
    ('dow', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
    
    ]
  """[
      ('Symbol', DataTypes.CATEGORICAL, InputTypes.ID),
      ('date', DataTypes.DATE, InputTypes.TIME),
      ('log_vol', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('open_to_close', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('days_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('week_of_year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('Region', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]"""

  def __init__(self,root_folder,klein=False,num_encoder_steps = 36,n_timesteps_forecasting=10, timeseries_interval = 6, input_t_dim = 60,num_epochs = 200, lr = 0.0001):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    self.pre_split_data_available = True
    self.klein = klein
    self.num_encoder_steps = num_encoder_steps
    self.n_timesteps_forecasting = n_timesteps_forecasting
    self.timeseries_interval = timeseries_interval
    self.input_t_dim= input_t_dim
    self.num_epochs = num_epochs
    self.lr = lr
    if klein:
      self.train_csv_path = root_folder+'/data/kidfail/klein_itd_60_ntsf40_ti6h/train_kidfail.csv'
      self.valid_csv_path = root_folder+'/data/kidfail/klein_itd_60_ntsf40_ti6h/valid_kidfail.csv'
      self.test_csv_path = root_folder+'/data/kidfail/klein_itd_60_ntsf40_ti6h/test_kidfail.csv'
      #self.train_csv_path = root_folder+'/data/kidfail/train_kidfail_5d8e1a34_e6140289.csv'
      #self.valid_csv_path = root_folder+'/data/kidfail/valid_kidfail_5d8e1a34_e6140289.csv'
      #self.test_csv_path = root_folder+'/data/kidfail/test_kidfail_5d8e1a34_e6140289.csv'
    else:
      
      tft_path = root_folder+f'/data/kidfail'
      output_folder = os.path.join(tft_path,f'itd_{input_t_dim}_ntsf{n_timesteps_forecasting}_ti{timeseries_interval}h')
      self.train_csv_path = os.path.join(output_folder,"train_kidfail.csv")
      self.valid_csv_path = os.path.join(output_folder,"valid_kidfail.csv")
      self.test_csv_path = os.path.join(output_folder,"test_kidfail.csv")
      
    
    # %%
    #import pandas as pd
    #train = pd.read_csv('/home/alexander/time_fusion_transformer/tft_outputs/data/kidfail/train_kidfail_5d8e1a34_e6140289.csv', index_col=0)
    #train['icd_17_9x'] = train['icd_17_9x'].fillna(0) 
    #train.to_csv('/home/alexander/time_fusion_transformer/tft_outputs/data/kidfail/train_kidfail_5d8e1a34_e6140289xx.csv')
    #pd.unique(train['icd_17_9x'])
    # %%


  def load_splitted_data(self):
    
    # %%
    train = pd.read_csv(self.train_csv_path, index_col=0)
    valid = pd.read_csv(self.valid_csv_path, index_col=0)
    test = pd.read_csv(self.test_csv_path, index_col=0)

    # %%
    
    self.set_scalers(train)
    return (self.transform_inputs(data,name) for data,name in zip([train, valid, test],["train", "valid", "test"]))
  
  def split_data(self, df):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')
   
   
    df = df.sort_values('id_combined')
    n_cases = df.shape[0]
    
    train_cutoff = int(n_cases*0.7)
    val_cutoff = int(n_cases*0.9)
    
    train = df.iloc[:train_cutoff]
    valid = df.iloc[train_cutoff:val_cutoff]
    test = df.iloc[val_cutoff:]
    

    self.set_scalers(train)
    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')
    
    #from util.general_util import dev_pickle
    #dev_pickle((self, df),"set_scalers")
    #(self, df) = dev_pickle(False,"set_scalers")
    
    column_definitions = self.get_column_definition()
    
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)
    
    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    data = df[real_inputs].values
    self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
    #self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
    #    df[[target_column]].values)  # used for predictions
    
    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str).fillna('other')
      categorical_scalers[col] = sklearn.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                 unknown_value=-1).fit(
          srs.values.reshape(-1, 1))
      
      
      num_classes.append(srs.nunique())
    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df,name):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    # %%
    #from tft_tf2.util.general_util import dev_pickle
    #dev_pickle((self, df,name),"transform_inputs")
    #from util.general_util import dev_pickle
    #dev_pickle((self, df),"transform_inputs")
    #(self, df,name) = dev_pickle(False,"transform_inputs")
    # %%
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()
    
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    
    
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    # %%
    real_inputs
    categorical_inputs
    # %%
    # Format real inputs
    output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df.values.reshape(-1, 1))

      # if any categorie was not present in test or val, it is replaced by the category for other
      # this is to prevent the embedding layers from bugging, since they, too make a mapping of knowns,
      # and they only know what was present in train, which cannot be -1. #So the fill value for missing category need to be filled with the category for other
      if not 0==(output[col]==-1).sum():
        code_of_other = self._cat_scalers[col].transform(np.array(['other']).reshape(-1, 1))[0][0]
        
        try:
          assert code_of_other!=-1
        except:
          print("other code was -1 ", col)
          from tft_tf2.util.general_util import dev_pickle
          dev_pickle((self, df,name),"transform_inputs")
        output[col] = output[col].replace(-1,code_of_other)
    
    
    for col in categorical_inputs:  
      assert 0==(output[col]==-1).sum()
          
      assert 0==output[col].isna().sum()
    
    for col in categorical_inputs:  
      try:
        #print(col,(output[col]==-1).sum())
        assert 0==(output[col]==-1).sum()
        
        assert 0==output[col].isna().sum()
      except:
        from util.general_util import dev_pickle
        dev_pickle((self, df,name),"transform_inputs")
        assert 0==(output[col]==-1).sum()
        
        assert 0==output[col].isna().sum()
     
    output = output.fillna(0)
    
    # category 4 gets assigned to missing values, so we need to remove them by imputing 0
    output['icd_17_9x'] = output['icd_17_9x'].replace(4,0)
    #pd.unique(output['icd_17_9x'])
    # %%
    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns

    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        #output[col] = self._target_scaler.inverse_transform(predictions[col].values.reshape(-1, 1))
        output[col] = predictions[col].values.reshape(-1, 1)

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': self.input_t_dim+self.n_timesteps_forecasting,
        'num_encoder_steps': self.num_encoder_steps,
        'n_timesteps_forecasting':self.n_timesteps_forecasting,
        'num_epochs': self.num_epochs,
        'early_stopping_patience': 8,
        'multiprocessing_workers': 5,
    }
    

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.3,
        'hidden_layer_size': 128,
        'learning_rate': self.lr,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 1,
        'stack_size': 1
    }
    
    if self.klein:
      model_params.update({'hidden_layer_size': 16})

    return model_params
