import pandas as pd import torch.nn as nn

class CNN(nn.Module):
def  init (self, K): super(CNN, self). init ()
self.conv_layers = nn.Sequential( # conv1
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
nn.ReLU(), nn.BatchNorm2d(32),
nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
# conv2
nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
nn.ReLU(), nn.BatchNorm2d(64),
nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
# conv3
nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
nn.ReLU(), nn.BatchNorm2d(128),
nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
# conv4
nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
nn.ReLU(),
nn.BatchNorm2d(256),
nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2),
)
self.dense_layers = nn.Sequential( nn.Dropout(0.4), nn.Linear(50176, 1024), nn.ReLU(),
nn.Dropout(0.4), nn.Linear(1024, K),
)
def forward(self, X):
out = self.conv_layers(X)
# Flatten
out = out.view(-1, 50176)
# Fully connected
out = self.dense_layers(out)
return out
idx_to_classes = {0: 'Apple  Apple_scab',
1: 'Apple  Black_rot',
2: 'Apple  Cedar_apple_rust',
3: 'Apple  healthy',
4: 'Background_without_leaves',
5: 'Blueberry  healthy',
6: 'Cherry  Powdery_mildew',
7: 'Cherry  healthy',
8: 'Corn  Cercospora_leaf_spot Gray_leaf_spot',
9: 'Corn  Common_rust',
10: 'Corn  Northern_Leaf_Blight',
11: 'Corn  healthy',
12: 'Grape  Black_rot',
13: 'Grape  Esca_(Black_Measles)',
14: 'Grape  Leaf_blight_(Isariopsis_Leaf_Spot)',
15: 'Grape  healthy',
16: 'Orange  Haunglongbing_(Citrus_greening)',
17: 'Peach  Bacterial_spot',
18: 'Peach  healthy',
19: 'Pepper,_bell  Bacterial_spot',
20: 'Pepper,_bell  healthy',
21: 'Potato  Early_blight',
22: 'Potato  Late_blight',
23: 'Potato  healthy',
24: 'Raspberry  healthy',
25: 'Soybean  healthy',
26: 'Squash  Powdery_mildew',
27: 'Strawberry  Leaf_scorch',
28: 'Strawberry  healthy',
29: 'Tomato  Bacterial_spot',
30: 'Tomato  Early_blight',
31: 'Tomato  Late_blight',
32: 'Tomato  Leaf_Mold',
33: 'Tomato  Septoria_leaf_spot',
34: 'Tomato  Spider_mites Two-spotted_spider_mite',
35: 'Tomato  Target_Spot',
36: 'Tomato  Tomato_Yellow_Leaf_Curl_Virus',
37: 'Tomato  Tomato_mosaic_virus',
38: 'Tomato  healthy'}
