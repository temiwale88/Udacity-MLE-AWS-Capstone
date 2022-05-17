import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import boto3

import sagemaker
import os
import logging
import sys
from pickle import dump
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sagemaker import get_execution_role
# session = sagemaker.Session()


os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

sagemaker_session = sagemaker.Session(boto3.session.Session(region_name='us-east-1'))
bucket = sagemaker_session.default_bucket()
s3 = boto3.client('s3')

from smdebug.profiler.utils import str2bool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion): 
        
    model.eval() #setting the model in evaluation mode
    running_correct = 0
    running_loss = 0.0
            
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Record the correct predictions for training data  
            running_loss += loss.item() * inputs.size(0) # Accumulate the loss 
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum() 

        # Record the testing loss and accuracy
        total_loss = running_loss // len(test_loader) #'//' is floor division
        total_acc = running_correct.double() // len(test_loader)
        
        logger.info(f"Testing Loss: {total_loss}")
        logger.info(f"Testing Accuracy: {total_acc}")

def train(model, train_loader, validation_loader, criterion, optimizer, epochs):
    """Here we'll train and examine trained models on validation dataset
    We'll keep training until we've exhausted our number of epochs
    We also want our class names to use as we decode predictions 
    For the validation phase, we will skip gradient and loss calculations. We'll also want our computed gradients of the model from the training phase so we will not zero_grad in the 'valid' phase
    ---- FUTURE EDITS: (1) Consider early stopping | AWS has early stopping in it's training jobs; """
    best_loss=1e6 # see here to understand this: https://bit.ly/3ldA2sK & https://bit.ly/3CVqZF3
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    pickle_path = "/opt/ml/output/"
    class_pickle_filename = "dog_breeds_labels.pkl"
    saved_data = os.path.join(pickle_path, class_pickle_filename)
    class_names = [item[4:].replace("_", " ") for item in train_loader.dataset.classes]
    
    print("Printing and saving classes: \n")
    print(class_names[:5])
    print(f"Saving to this directory: {pickle_path}")
    
    # save as a pickle file
    with open(saved_data, 'wb') as f:
        dump(class_names, f)

    # Save it to s3 | see boto3 docs here: https://bit.ly/3IQXCFh
    with open(saved_data, 'rb') as data:
        s3.upload_fileobj(data, bucket, f'dogImages/classes/{class_pickle_filename}')

    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch+1}")
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss: # see here to understand this: https://bit.ly/3ldA2sK & https://bit.ly/3CVqZF3
                    best_loss=epoch_loss
                else:
                    loss_counter+=1


            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                                 epoch_loss,
                                                                                 epoch_acc,
                                                                                 best_loss))
        if loss_counter==1: #counter=1 IF our best loss is bigger than 1e6 (1 million): BAD
            break
        if epoch==0:
            break
    return model

# Per documentation let's try Squeezenet: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False #This is the key part for transfer learning as we freeze the convolutional layers before we add our layer. Assumes that we set 'feature_extracting' = True for grad to be set to 'False'. See PyTorch documentation on transfer learning
            
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def main(args):

    model=net()
    print(model) #let's understand our model architecture

    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion =  nn.CrossEntropyLoss(ignore_index=133) #ignore last non-useful index?
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
     
    # Download the data from s3
    train_key_prefix = "dogImages/train"
    train_folder_name = './dogImages/train'
    
    if not os.path.exists(train_folder_name):
        os.makedirs(train_folder_name)
        
    sagemaker_session.download_data(train_folder_name, bucket, train_key_prefix)
    
    test_key_prefix = "dogImages/test"
    test_folder_name = './dogImages/test'
    
    if not os.path.exists(test_folder_name):
        os.makedirs(test_folder_name)
    
    validation_key_prefix = "dogImages/valid"
    validation_folder_name = './dogImages/valid'
    
    if not os.path.exists(validation_folder_name):
        os.makedirs(validation_folder_name)
        
    sagemaker_session.download_data(train_folder_name, bucket, train_key_prefix)
    sagemaker_session.download_data(test_folder_name, bucket, test_key_prefix)
    sagemaker_session.download_data(validation_folder_name, bucket, validation_key_prefix)
    
    # DataLoaders and Datasets
    # From PyTorch's documentation on transfer learning
    # Data augmentation and normalization for training
    # Just normalization for validation and testing


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
    }
    
    
    # See helper codes from skysign on how to load this data: https://bit.ly/3MNdKe0

    train_data = datasets.ImageFolder(train_folder_name, transform=data_transforms['train'])
    validation_data = datasets.ImageFolder(validation_folder_name, transform=data_transforms['test']) #using the same transformation as in test set
    test_data = datasets.ImageFolder(test_folder_name, transform=data_transforms['test'])
    
    batch_size=args.batch_size
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, 
                                               num_workers=2,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=batch_size, 
                                            #    num_workers=2,
                                               shuffle=False)

    validation_loader = torch.utils.data.DataLoader(validation_data,
                                               batch_size=batch_size, # same as test
                                            #    num_workers=2,
                                               shuffle=False)

    print("Initializing Datasets and Dataloaders...")
    
    epochs = args.epochs
    
    logger.info("Starting Model Training")
    train(model, train_loader, validation_loader, criterion, optimizer, epochs)
    
    logger.info("Testing Model")
    test(model, test_loader, criterion)
        

    logger.info("Printing and saving model dict: \n")
    logger.info(model.state_dict())
    logger.info(f"Saving to this model directory: {os.environ['SM_MODEL_DIR']}")
    
    # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, 'dogs_classification_hpo.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__=='__main__':
 
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--num_classes", type=int, default=True)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
  
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50, # change as needed
        metavar="N",
        help="number of epochs to train (default: ?)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    main(args)
