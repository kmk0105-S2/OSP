import torch.nn as nn

# 1x1 convolution
def conv1x1(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return model


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return model

###########################################################################
# Question 1 : Implement the "bottle neck building block" part.
# Hint : Think about difference between downsample True and False. How we make the difference by code?
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        self.layer = nn.Sequential(
            conv1x1(in_channels, middle_channels, stride, 0),
            conv3x3(middle_channels, middle_channels, 1, 1),
            conv1x1(middle_channels, out_channels, 1, 0)
        )

        if downsample or in_channels != out_channels:
            self.make_equal_channel = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.make_equal_channel = nn.Identity()

    def forward(self, x):
        identity = self.make_equal_channel(x)
        out = self.layer(x)
        return out + identity
###########################################################################



###########################################################################
# Question 2 : Implement the "class, ResNet50_layer4" part.
# Understand ResNet architecture and fill in the blanks below. (25 points)
# (blank : #blank#, 1 points per blank )
# Implement the code.
class ResNet50_layer4(nn.Module):
    def __init__(self, num_classes= 10 ): # Hint : How many classes in Cifar-10 dataset?
        super(ResNet50_layer4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                # Hint : Through this conv-layer, the input image size is halved.
                #        Consider stride, kernel size, padding and input & output channel sizes.
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, 256, downsample=True),
            ResidualBlock(256, 64, 256, downsample=False),
            ResidualBlock(256, 64, 256, downsample=False)
        )
        self.layer3 = nn.Sequential(
            ##########################################
            ############# fill in here (20 points)
            ResidualBlock(256, 128, 512, downsample=True),
            ResidualBlock(512, 128, 512, downsample=False),
            ResidualBlock(512, 128, 512, downsample=False),
            ResidualBlock(512, 128, 512, downsample=False)
            ####### you can refer to the 'layer2' above
            #########################################
        )
        self.layer4 = nn.Sequential(
            ##########################################
            ############# fill in here (20 points)
            ResidualBlock(512, 256, 1024, downsample=True),
            ResidualBlock(1024, 256, 1024, downsample=False),
            ResidualBlock(1024, 256, 1024, downsample=False),
            ResidualBlock(1024, 256, 1024, downsample=False),
            ResidualBlock(1024, 256, 1024, downsample=False),
            ResidualBlock(1024, 256, 1024, downsample=False)
            ####### you can refer to the 'layer2' above
            #########################################
        )

        self.fc = nn.Linear(1024, 10)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)

        return out
###########################################################################