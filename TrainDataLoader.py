from torch.utils.data import DataLoader
import TrainData
import InstanceData
import SemanticData
from torch.utils.tensorboard import SummaryWriter

#dota1.0 原始图片及txt标注
root_dir = "E:/研究生论文/项目1/图像数据/DOTA1.0/Train/images"
label_dir = "E:/研究生论文/项目1/图像数据/DOTA1.0/Train/labelTxt-v1.0/labelTxt"
#ISAID 数据集，instance_mask, semantic_mask
instance_dir = "E:\\研究生论文\\项目1\\图像数据\\ISAID\\train\\instance_mask\\images"
semantic_dir = "E:\\研究生论文\\项目1\\图像数据\\ISAID\\train\\Semantic_masks\\images"

#获取 dataset
train_dataset = TrainData.TrainDataSet(root_dir, label_dir)
instance_dataset = InstanceData.InstanceDataSet(instance_dir)
semantic_dataset = SemanticData.SemanticDataSet(semantic_dir)



#DOTA1.0 的dataloader
train_dataloader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=False,num_workers=0)
writer = SummaryWriter("./logs")
step = 0
for data in train_dataloader:
    imgs, label = data
    print(imgs.shape)
    writer.add_images('train1', imgs, step)
    step += 1
writer.close()

#instance_mask 的 dataloader
instance_dataloader = DataLoader(dataset=instance_dataset,batch_size=64,shuffle=False,num_workers=0)
writer = SummaryWriter("./logs")
step = 0
for data in instance_dataloader:
    instance_imgs, instance_labels = data
    print(instance_imgs.shape)
    writer.add_images('instance1',instance_imgs,step)
    step+=1
writer.close()


writer = SummaryWriter("./logs")
step = 0
# semantic_mask 的 dataloader
semantic_dataloader = DataLoader(dataset=semantic_dataset,batch_size=64,shuffle=False,num_workers=0)
for data in semantic_dataloader:
    semantic_imgs, semantic_labels = data
    print(semantic_imgs.shape)
    writer.add_images('semantic1', semantic_imgs, step)
    step += 1
writer.close()





