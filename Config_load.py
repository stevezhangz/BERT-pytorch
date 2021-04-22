# encode utf-8
# code by steve zhang z
# Time: 4/22/2021
# electric address: stevezhangz@163.com
import configparser
conf=configparser.ConfigParser()
conf.read("Config.cfg")
maxlen = int(conf.get("hyper_parameters","maxlen"))
batch_size = int(conf.get("hyper_parameters","batch_size"))
max_pred = int(conf.get("hyper_parameters","max_pred"))
n_layers = int(conf.get("hyper_parameters","n_layers"))
n_heads = int(conf.get("hyper_parameters","n_heads"))
d_model = int(conf.get("hyper_parameters","d_model"))
d_ff =eval(conf.get("hyper_parameters","d_ff"))
d_k = int(conf.get("hyper_parameters","d_k"))
d_v=int(conf.get("hyper_parameters","d_v"))
n_segments = int(conf.get("hyper_parameters","n_segments"))
random_seed=int(conf.get("hyper_parameters","seed"))
lr=float(conf.get("hyper_parameters","lr"))
epoches=int(conf.get("hyper_parameters","epc"))
data_dir=conf.get("file_dir","data")
weight_dir=conf.get("file_dir","data")
device=conf.get("device","force")
default_tensor_type=conf.get("device","tensor_dtype")
use_gpu=int(conf.get("device","confirm_gpu"))
drop=float(conf.get("hyper_parameters","drop"))
