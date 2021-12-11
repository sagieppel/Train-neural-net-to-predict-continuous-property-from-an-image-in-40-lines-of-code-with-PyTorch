import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
Learning_Rate=1e-5
width=height=900 # image width and height
batchSize=1
#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(): # First lets load random image and  the corresponding annotation
    FillLevel=np.random.random()
    Img=np.zeros([900,900,3],np.uint8)
    Img[0:int(FillLevel*900),:]=255
    transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    Img=transformImg(Img)
    return Img,FillLevel
#--------------Load batch of images-----------------------------------------------------
def LoadBatch(): # Load batch of images
    images = torch.zeros([batchSize,3,height,width])
    FillLevel = torch.zeros([batchSize])
    for i in range(batchSize):
        images[i],FillLevel[i]=ReadRandomImage()
    return images,FillLevel
#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.resnet18(pretrained=True) # Load net
Net.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
Net = Net.to(device)
optimizer = torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer
#----------------Train--------------------------------------------------------------------------
AverageLoss=np.zeros([50]) # Save average loss for display
for itr in range(500001): # Training loop
   images,GTFillLevel=LoadBatch() # Load taining batch
   images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
   GTFillLevel = torch.autograd.Variable(GTFillLevel, requires_grad=False).to(device) # Load annotation
   PredLevel=Net(images) # make prediction
   Net.zero_grad()
   Loss=torch.abs(PredLevel-GTFillLevel).mean()
   Loss.backward() # Backpropogate loss
   optimizer.step() # Apply gradient descent change to weight
   AverageLoss[itr%50]=Loss.data.cpu().numpy() # Save loss average
   print(itr,") Loss=",Loss.data.cpu().numpy(),'AverageLoss',AverageLoss.mean()) # Display loss
   if itr % 1000 == 0: # Save model
        print("Saving Model" +str(itr) + ".torch") #Save model weight
        torch.save(Net.state_dict(),   str(itr) + ".torch")
