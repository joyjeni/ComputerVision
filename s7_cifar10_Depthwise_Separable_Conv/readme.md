### Contributors
Chethan kumar V  
Vivek K  
Jenisha Thankaraj 



#### CIFAR10

Achieve 85% accuracy. Total Params to be less than 200k

#### 1. Architecture
#### Model 1

[1]  Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          18,432
       BatchNorm2d-6           [-1, 64, 32, 32]             128
              ReLU-7           [-1, 64, 32, 32]               0
           Dropout-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 32, 16, 16]           2,048
           Conv2d-10           [-1, 64, 14, 14]             576
      BatchNorm2d-11           [-1, 64, 14, 14]             128
             ReLU-12           [-1, 64, 14, 14]               0
          Dropout-13           [-1, 64, 14, 14]               0
           Conv2d-14            [-1, 128, 7, 7]           8,192
      BatchNorm2d-15            [-1, 128, 7, 7]             256
             ReLU-16            [-1, 128, 7, 7]               0
          Dropout-17            [-1, 128, 7, 7]               0
           Conv2d-18          [-1, 128, 11, 11]         147,456
      BatchNorm2d-19          [-1, 128, 11, 11]             256
             ReLU-20          [-1, 128, 11, 11]               0
          Dropout-21          [-1, 128, 11, 11]               0
           Conv2d-22            [-1, 128, 6, 6]         147,456
      BatchNorm2d-23            [-1, 128, 6, 6]             256
             ReLU-24            [-1, 128, 6, 6]               0
          Dropout-25            [-1, 128, 6, 6]               0
        AvgPool2d-26            [-1, 128, 1, 1]               0
           Conv2d-27            [-1, 128, 1, 1]          16,384
             ReLU-28            [-1, 128, 1, 1]               0
      BatchNorm2d-29            [-1, 128, 1, 1]             256
          Dropout-30            [-1, 128, 1, 1]               0
           Conv2d-31             [-1, 10, 1, 1]           1,280
================================================================
Total params: 344,032
Trainable params: 344,032
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.25
Params size (MB): 1.31
Estimated Total Size (MB): 5.58
----------------------------------------------------------------

[2] Used dilated kernels instead of stride \\

[3] No maxpooling layer add stride=2 instead
```python
 self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(1, 1), padding=0, bias=False,stride=2),
        )  # 
```
#### 2. Receptive Field Calculation > 52

#### 3. Add two Depthwise Seperable Convolution
```python
self.depthwise1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=0, groups=32, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) 
```
#### 4. Add Dilated Kernel
# CONVOLUTION BLOCK 3
```python
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=4, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  #
        
  ```
        
#### 5. Add Global Average Pooling

```python      # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # output_size = 1

```

#### 6. Add augumentation using Albumentation library and apply:
[1] horizontal flip

[2]  shiftScaleRotate

[3] coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)


