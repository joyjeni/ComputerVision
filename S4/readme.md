### Training MNIST data

1. 99.6% validation accuracy
2. Less than 20k Parameters
3. Used Batch Normalization BN
4. Dropout
5. Fully connected layer 

#### 1. Load dataset

```python
train = torchvision.datasets.MNIST('./var', train=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test = torchvision.datasets.MNIST('./var', train=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=True)

```
<a href="https://imgur.com/y6ct1fm"><img src="https://i.imgur.com/y6ct1fm.png" title="source: imgur.com" /></a>

### 2. Build Network

```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.069)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.069)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.069)
        )
       
        self.fc = nn.Sequential(
            
            nn.Linear(128, 10)
        )

            
        
     
                
        
    def forward(self, x):
        x = self.conv1(x)
        self.after_conv1 = x
        x = self.conv2(x)
        self.after_conv2 = x
        
        x = self.conv3(x)
        self.after_conv3=x
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
       # x=F.adaptive_avg_pool2d(x,(10,10))
        
        x = F.log_softmax(x, dim=1)
        return x
        
 ```

### 3. Train Network

```python
for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        log_ps = model(images)
        
        ps = torch.exp(log_ps)                
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

        loss = loss_function(log_ps, labels)
        loss.backward()
        optimizer.step()
 ```     
     
 ### 4. Validate Network
 
 ```python
  accuracy=0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss += loss_function(log_ps, labels)  # sum up batch loss
            pred = log_ps.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))   
            
            
 ```
     
### 5. Visualize 

```python
# plot training history
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
ax = plt.gca()
ax.set_xlim([0, epoch + 2])
plt.ylabel('Loss')
plt.plot(range(1, epoch + 2), train_losses[:epoch+1], 'g', label='Training Loss')
plt.plot(range(1, epoch + 2), test_losses[:epoch+1], 'b', label='Validation Loss')
ax.grid(linestyle='-.')
plt.legend()
plt.subplot(2,1,2)
ax = plt.gca()
ax.set_xlim([0, epoch+2])
plt.ylabel('Accuracy')
plt.plot(range(1, epoch + 2), train_accu[:epoch+1], 'g', label='Training Accuracy')
plt.plot(range(1, epoch + 2), test_accu[:epoch+1], 'b', label='Validation Accuracy')
ax.grid(linestyle='-.')
plt.legend()
plt.show()       
```

<a href="https://imgur.com/6RBZy3h"><img src="https://i.imgur.com/6RBZy3h.png" title="source: imgur.com" /></a>






