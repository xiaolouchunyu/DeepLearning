"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
 
           
 
        
        self.features = models.alexnet(pretrained = True).features
        
        
        self.classifier = nn.Sequential(
            
            nn.Conv2d(3, 30, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(30, 60, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(60, 120, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(120, 240, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(240, 30, kernel_size=1, padding=0),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # (32, 120, 120)
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # (32, 240, 240)
            
            
            nn.Conv2d(30, num_classes, kernel_size=1, padding=0
                      
                     )
            
            
         
            
            
        #nn.Conv2d(256, 4096, kernel_size=1, padding=0),
        #nn.ReLU(),
        #nn.Dropout(),
        #nn.MaxPool2d(2, 2),
        #nn.Dropout2d(p=0.2),
        #nn.Conv2d(2048, 4096, kernel_size=3, padding=1),
        #nn.ReLU(),
        #nn.Dropout(),
        #nn.MaxPool2d(2, 2),
        #nn.Dropout2d(p=0.2),
        #nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        #nn.Upsample(scale_factor = 40),
        #nn.Conv2d(512, 2048, kernel_size=3, padding=1),
        #nn.ReLU(),
        #nn.MaxPool2d(2, 2),
        #nn.Dropout2d(p=0.2),
        #nn.Conv2d(4096, num_classes, kernel_size=1, padding=0),
        #nn.ReLU(),
        #nn.Upsample(scale_factor=40),
        #nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
      
        #nn.MaxPool2d(2, 2),
        #nn.Dropout2d(p=0.2),
        )
       # self.fc = nn.Sequential(
       ##      nn.Linear(4096 , 2048),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2048, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(128, num_classes),
         #    nn.Tanh()
        #  )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.features(x)
        x = self.classifier(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
