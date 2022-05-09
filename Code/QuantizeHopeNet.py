from modules.hopenet import Hopenet
from torchvision import models
from frames_dataset import FramesDataset
import yaml
from torch.utils.data import DataLoader
import torch
import os
import timeit
import numpy as np

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# Creating a Dynamic Quant Model only works for Linear layers and LTSMs
def dynamic_quantization(model):
    model_int8 = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Conv2d, hopenet.layer1, hopenet.layer2, hopenet.layer3, hopenet.layer4, hopenet.fc_yaw,
         hopenet.fc_roll, hopenet.fc_pitch, hopenet.fc_finetune, torch.nn.BatchNorm2d, hopenet.conv1,
         torch.nn.Linear
         },
        dtype=torch.qint8)  # the target dtype for quantized weights
    return model_int8

# Static quantization must add quant layers in the forward training loop.
def static_quantization(model):
    print('Hello world')


def time_it(model, input):
    time_int = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for idx in range(100):
        start.record()
        model(input)
        end.record()
        torch.cuda.synchronize()
        time_int.append(start.elapsed_time(end))
    return np.mean(time_int)


def create_mid_fuses():
    mid_fuses = []
    for layer_num, bottneck_num in enumerate([3, 4, 6, 3]):
        for bottleneck in range(bottneck_num):
            for i in range(2):
                current_fuse = []
                for module in ['conv', 'bn']:
                    current_fuse.append(f'layer{layer_num + 1}.{bottleneck}.{module}{i + 1}')
                mid_fuses.append(current_fuse)

    return mid_fuses



def create_end_fuses():
    end_fuses = []
    for layer_num, bottneck_num in enumerate([3, 4, 6, 3]):
        for bottleneck in range(bottneck_num):
            current_fuse = []
            for module in ['conv', 'bn']:
                current_fuse.append(f'layer{layer_num + 1}.{bottleneck}.{module}3')
            current_fuse.append(f'layer{layer_num + 1}.{bottleneck}.relu')
            end_fuses.append(current_fuse)

    return end_fuses

def load_config(config_path):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def inference_on_quantized(model, input):
    #input_quant = torch.quantize_per_tensor(input, 0.1, 10, torch.quint8)
    # The above quant_per_tensor should be sone with the self.quant call
    return model.forward(input)

def get_input_for_quant(config):
    # Load our data
    dataset = FramesDataset(is_train= True, **config['dataset_params'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=True)
    for x in dataloader:
        input = x['driving']
    return input

'''
We define a bottleneck architecture as the type found in the ResNet paper where [two 3x3 conv layers] 
are replaced by [one 1x1 conv, one 3x3 conv, and another 1x1 conv layer].
https://stats.stackexchange.com/q/205150

https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
'''

if __name__ == '__main__':
    # Parameters taken from generator full, hopenet implementation
    hopenet = Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66, quant=True)


    # Load config for parameters for dataset loader
    config_path = r'./config/QuantizeConfig.yaml' # Path of Quantization
    config = load_config(config_path)

    # Get input
    input = get_input_for_quant(config)

    # Static Quantization runs on cpu
    device = 'cpu'
    hopenet.to(device)
    input.to(device)

    '''Load the hopenet checkpoint'''
    train_params = config['train_params']
    hopenet_state_dict = torch.load(train_params['hopenet_snapshot'])
    hopenet.load_state_dict(hopenet_state_dict)

    # model must be set to eval mode for static quantization logic to work
    hopenet.eval()

    '''Experiment with configuration engine'''
    # hopenet.qconfig = torch.quantization.get_default_qconfig('qnnpack') # Configure for inference on ARM
    hopenet.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # For x86 CPUs

    # Create fuses for model
    all_fuses = create_end_fuses()  + create_mid_fuses()
    hopenet_fused = torch.quantization.fuse_modules(hopenet, [['conv1', 'bn1', 'relu']] + all_fuses)
    hopenet_prepared = torch.quantization.prepare(hopenet_fused)

    # Calibrate the quantization
    hopenet_prepared(input)

    # Convert
    hopenet_int8 = torch.quantization.convert(hopenet_prepared)
    hopenet_int8.eval()

    '''Check Memory and computational differences'''
    f = print_size_of_model(hopenet,"fp32")
    q = print_size_of_model(hopenet_int8,"int8")
    print("{0:.2f} times smaller".format(f/q))

    # # compare the performance
    print("Floating point FP32")
    time_float = time_it(hopenet, input)
    print(f'Time of: {time_float}\n')

    print("Quantized INT8")
    time_int = time_it(hopenet_int8, input)
    print(f'Time of: {time_int}\n')

    print("{0:.2f} times faster".format(time_float / time_int))
