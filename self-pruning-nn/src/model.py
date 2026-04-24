import torch  # type: ignore
import torch.nn as nn  # type: ignore
from layers import PrunableLinear  # type: ignore

class PrunableCIFAR10Net(nn.Module):
    """
    A Convolutional Neural Network (CNN) for CIFAR-10 classification,
    designed to rigorously reach ~90% accuracy iteratively mapping features
    and utilizing PrunableLinear layers within the final classifier logically.
    """
    def __init__(self):
        super(PrunableCIFAR10Net, self).__init__()
        
        # VGG-style Deep Feature Extractor (CNN)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 16 x 16
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128 x 8 x 8
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 256 x 4 x 4
        )
        
        # Classifier Head mapping our explicitly Custom PrunableLinear layers computationally
        self.fc_input_features = 256 * 4 * 4  # 4096 params functionally
        
        self.classifier = nn.Sequential(
            PrunableLinear(self.fc_input_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            PrunableLinear(512, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten logic effectively mapping boundaries statically
        x = self.classifier(x)
        return x
