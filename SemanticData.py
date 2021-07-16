import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image



class SemanticDataSet(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir  # 照片所在根目录
        self.img_path = os.listdir(self.root_dir)  # 获取图片list

    def __getitem__(self, idx):
        img_item_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_item_name)
        img = Image.open(img_item_path)
        #将PIL类型图片转成Tensor
        trans_tensor = torchvision.transforms.ToTensor()
        trans_resize = torchvision.transforms.Resize((128, 128))
        trans_compose = torchvision.transforms.Compose([trans_tensor, trans_resize])
        img_tensor = trans_compose(img)

        # Instance与Semantic 的label都设置成对应的DOTA1.0图片的文件名
        label = img_item_name.split('_')[0]+".png"

        return img_tensor, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':

    root_dir = "E:\\研究生论文\\项目1\\图像数据\\ISAID\\train\\Semantic_masks\\images"

    train_dataset = SemanticDataSet(root_dir)
    img, label = train_dataset[0]

    print(type(img))