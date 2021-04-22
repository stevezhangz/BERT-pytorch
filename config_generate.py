import configparser

# param
maxlen = "120"
batch_size = "12"
max_pred = "5"
n_layers = "6"
n_heads = "12"
d_model = "1000"
d_ff = "1000*4"
d_k = "100"
d_v = "100"
n_segments = "2"
seed = "9527"
lr = "0.0001"
epc = "250"
drop="0.5"

#dir
data_dir="./data"
weight_dir="./checkpoint"
Config_file="Config.cfg"

# device
device="cuda:0"
default_tensor_type="torch.cuda.FloatTensor"
use_gpu="1"

# generate configs
conf=configparser.ConfigParser()
cfg=open(Config_file,"w")
conf.add_section("hyper_parameters")
conf.set("hyper_parameters","maxlen",maxlen)
conf.set("hyper_parameters","batch_size",batch_size)
conf.set("hyper_parameters","max_pred",max_pred)
conf.set("hyper_parameters","n_layers",n_layers)
conf.set("hyper_parameters","n_heads",n_heads)
conf.set("hyper_parameters","d_model",d_model)
conf.set("hyper_parameters","d_ff",d_ff)
conf.set("hyper_parameters","d_k",d_k)
conf.set("hyper_parameters","d_v",d_v )
conf.set("hyper_parameters","n_segments",n_segments)
conf.set("hyper_parameters","seed",seed)
conf.set("hyper_parameters","lr",lr)
conf.set("hyper_parameters","epc",epc)
conf.set("hyper_parameters","drop",drop)

conf.add_section("file_dir")
conf.set("file_dir","data",data_dir)
conf.set("file_dir","data",weight_dir)

conf.add_section("device")
conf.set("device","force",device)
conf.set("device","tensor_dtype",default_tensor_type)
conf.set("device","confirm_gpu",use_gpu)
conf.write(cfg)
cfg.close()
