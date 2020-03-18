import torch

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels=None, transform=None, self_mix=False):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.self_mix = self_mix

    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.transform(image=self.imgs[idx])['image']
            if self.self_mix:
                img2 = self.transform(image=self.imgs[idx])['image']
                img = 0.5 * img + 0.5 * img2

        else:
            img = self.imgs[idx]

        if self.labels is not None:
            return img, self.labels[idx]
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def get_dataloader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
