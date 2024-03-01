import torch
import torchvision.models as models
from torch import nn


def build_model(pretrainedPath, num_classes, args):
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    print('[INFO]: Loading pre-trained weights')
    checkpoint = torch.load(pretrainedPath, map_location=torch.device('cpu'))

    # rename moco pre-trained keys
    if args.isCheckpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint.state_dict()
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    # Change the final classification head.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features=num_classes)

    return model
