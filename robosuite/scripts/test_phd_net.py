from robosuite.scripts.training_phd_net import TransitionDataset
from torch.utils.data import DataLoader

from robosuite.mcts.tree_search import *
from robosuite.mcts.resnet import ResNet, BasicBlock, Bottleneck
from robosuite.mcts.inception import Inceptionv4

if __name__ == "__main__":
    model_name_list = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNext50', 'Wide_ResNet50', 'InceptionV4']
    model_name = model_name_list[0]
    model = ResNet(BasicBlock, [2, 2, 2, 2]).to('cuda')
    model.load_state_dict(torch.load('robosuite/networks/ ' +model_name +'_depth_img ' +'.pt'), strict=False)

    transition_dataset = TransitionDataset()
    val_len = transition_dataset.total_num

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    validation_loader = DataLoader(transition_dataset, batch_size=val_len)
