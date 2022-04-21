# %%
import os
import sys

#pathProject

try:
    os.chdir('/app')
    pathProject = '/app'
    
except:
    pathProject = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(pathProject)
print("pathProject",pathProject)

sys.path.append('tft_tf2')

from libs.tft_model import TemporalFusionTransformer
import pickle
import tensorflow.compat.v1 as tf
import libs.utils as utils

import expt_settings.configs



K = tf.keras.backend

tf.compat.v1.disable_eager_execution()
# %%
def recreate_TFT_with_param_path():#output_folder,experiment_name,name,use_gpu,klein=True):
    # %%
    experiment_name = 'main_cpu_itd_120_ntsf20_ti6_klein_True'
    #experiment_name = 'fixed_klein1'
    #experiment_name = 'main_klein_True'
    name = 'kidfail'
    output_folder = pathProject+f'/tft_outputs'
    klein = True
    use_gpu = False
    # %%
    ExperimentConfig = expt_settings.configs.ExperimentConfig
    config = ExperimentConfig(name, output_folder)  
    data_formatter = config.make_data_formatter()
    data_formatter = data_formatter(klein=klein,num_encoder_steps = 56,n_timesteps_forecasting=10, timeseries_interval = 6, input_t_dim = 60)
    # %%
    dir(config)
    config.root_folder
    config.model_folder
    # %%
    #train, valid, test = data_formatter.load_splitted_data()
    # %%
    #param_path = config.model_folder+f'/{experiment_name}/params_pickle.pkl'
    param_path = config.model_folder+f'/{experiment_name}/best_params_pickle.pkl'
    raw_params = pickle.load(open(param_path,'rb'))
    
    # %%
    #raw_params['n_timesteps_forecasting'] = raw_params['total_time_steps']-raw_params['num_encoder_steps']
    # %%
    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

   
    # %%
    tf.reset_default_graph()
    #sess = tf.Graph().as_default(),tf.Session(config=tf_config) 
    sess = tf.Session(graph = tf.Graph(),config=tf_config) 
    tf.keras.backend.set_session(sess)
    # Inputs.
    
    best_params = raw_params
    model = TemporalFusionTransformer(best_params, use_cudnn=use_gpu,for_prediction=True)
  
    # %%
    model.load(pathProject+f'/tft_outputs/saved_models/kidfail/{experiment_name}')
   
    # %%
    model.model.load_weights(config.model_folder+f'/{experiment_name}/best_weights.hdf5')
    # %%
    other_output_map = model.predict(test, return_targets=True)    
    

    # %%
    model.model.save('my_model')

# %%
recreate_TFT_with_param_path()