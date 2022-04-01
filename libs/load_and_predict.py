
# %%
from curses import raw
from tft_model import TemporalFusionTransformer
import pickle
import tensorflow.compat.v1 as tf
import utils
import os
import sys
import expt_settings.configs
K = tf.keras.backend
Lambda = tf.keras.layers.Lambda
tf.compat.v1.disable_eager_execution()

pathProject = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
pathProject
# %%
try:
    os.chdir(pathProject)
except:
    os.chdir('/app/')
    pathProject = '/app/'

sys.path.append('src')
# %%
def recreate_TFT_with_param_path(param_path,use_gpu):
    name = 'fixed_klein1'
    name = 'fixed'
    
    # %%
    ExperimentConfig = expt_settings.configs.ExperimentConfig
    config = ExperimentConfig('kidfail', '/home/alexander/time_fusion_transformer/tft_output')  
    data_formatter = config.make_data_formatter()
    data_formatter = data_formatter(klein=False)
    train, valid, test = data_formatter.load_splitted_data()
    # %%
    use_gpu = False
    param_path = pathProject+f'/tft_outputs/saved_models/kidfail/{name}/params_pickle.pkl'
    raw_params = pickle.load(open(param_path,'rb'))
    raw_params['n_time_steps_forecasting'] = raw_params['total_time_steps']-raw_params['num_encoder_steps']
    # %%
    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

    print("*** Running tests ***")
   
    # %%
    tf.reset_default_graph()
    #sess = tf.Graph().as_default(),tf.Session(config=tf_config) 
    sess = tf.Session(graph = tf.Graph(),config=tf_config) 
    tf.keras.backend.set_session(sess)
    # Inputs.
    
    best_params = raw_params
    model = TemporalFusionTransformer(best_params, use_cudnn=use_gpu,for_prediction=True)
  
    # %%
    model.load(f'/home/alexander/time_fusion_transformer/tft_outputs/saved_models/kidfail/{name}')
    # %%
    other_output_map = model.predict(test, return_targets=True)    
    

    # %%
    model.model.save('my_model')

# %%
