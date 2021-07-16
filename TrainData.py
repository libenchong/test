import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image


class TrainDataSet(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir #照片所在根目录
        self.label_dir = label_dir #图片label所在的根目录
        self.img_path = os.listdir(self.root_dir) #获取图片list
        self.label_path = os.listdir(self.label_dir)#获取图片label list

    def __getitem__(self, idx):
        img_item_name = self.img_path[idx]#获得一张图片的文件名
        img_item_path = os.path.join(self.root_dir,img_item_name)#拼装一张图像的绝对路径
        img = Image.open(img_item_path)#读取一张图像
        img = img.convert("RGB")#将不同通道数的图片全部装换成RGB 3通道图片，https://blog.csdn.net/missyougoon/article/details/85331493
        #将PIL类型的图像转成tensor
        trans_tensor = torchvision.transforms.ToTensor()
        trans_resize = torchvision.transforms.Resize((128, 128))#resize 图像
        trans_compose = torchvision.transforms.Compose([trans_tensor, trans_resize])
        img_tensor = trans_compose(img)

        #label文件的读取
        label_item_name = self.label_path[idx]
        label_item_path = os.path.join(self.label_dir, label_item_name)
        jy = open(label_item_path)
        label = jy.read()

        return img_tensor, label

    def __len__(self):
        return len(self.img_path)


if __name__=='__main__':

    root_dir = "E:/研究生论文/项目1/图像数据/DOTA1.0/Train/images"
    label_dir = "E:/研究生论文/项目1/图像数据/DOTA1.0/Train/labelTxt-v1.0/labelTxt"

    train_dataset = TrainDataSet(root_dir, label_dir)
    img, label= train_dataset[1]

    print(type(img))