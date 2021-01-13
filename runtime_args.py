'''
Configurations.
'''

import argparse




parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='Specify the model that you want to train.', required=True)
parser.add_argument('--data_folder', type=str, help='Specify the path to the folder where the data is.', required=True)
parser.add_argument('--model_save_path', type=str, help='Specify the path to save the model.', default='./saved_models/')
parser.add_argument('--epoch_feature', type=int, help='Specify the number of epochs for the training/testing for feature-level.', default=5)
parser.add_argument('--epoch_object', type=int, help='Specify the number of epochs for the training/testing for object-level.', default=10)
parser.add_argument('--batch_size', type=int, help='Specify the batch size to be used during training/testing.', default=10)
parser.add_argument('--learning_rate', type=float, help='Specify the batch size to be used during training/testing.', default=1e-4)
parser.add_argument('--num_workers', type=int, help='Specify the number of workers to be used to load the data.', default=4)
parser.add_argument('--shuffle', type=bool, help='Specify if the data for training/testing should be shuffled or not.', default=True)
parser.add_argument('--img_size', type=int, help='Specify the size of the input image. Please double check before entering a value.', default=64)
parser.add_argument('--device', type=str, help='Specify which device to be used for the evaluation. Either "cpu" or "gpu".', default='gpu')

args = parser.parse_args()