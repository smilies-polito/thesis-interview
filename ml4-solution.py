import torch
import torch.nn as nn
import torch.optim as optim
from tochvision import datasets, transforms

batch_size = 64
learning_rate = 0.001
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
								root = "./data",
								train= True,
								download= True,	
								transform = transform)

test_dataset = datasets.MNIST(
								root = "./data",
								train= True,
								transform = transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#model 
class SimpleNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(28*28, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self):
		x = x.view(x.size(0), -1)
		x = torch.relu(self.fc1(x))
		x = self.fc2(x)
		return x

model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

model.train()

# training
for epoch in range(epochs):
	running_loss = 0.0
	for images, labels in train_loader:
		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images) # forward pass

		loss = criterion(outputs, labels) # loss computation
		
		optimizer.zero_grad()
		loss.backward()

		optimizer.step()

		running_loss += loss.item()

# evaluation
model.eval()
correct, total = 0,0

with torch.no_grad():
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

accuracy = 100*correct/total




		



		
		

		
	
		

