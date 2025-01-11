import torch
from torch.utils.data import Dataset
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps




class MyDataSet(Dataset):
    def __init__(self, enc_inputs, dec_inputs,txt_attn):
        super(MyDataSet, self).__init__()
        self.image = enc_inputs
        self.text = dec_inputs
        self.txt_attn = txt_attn
    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return self.image[idx], self.text[idx],self.txt_attn[idx]

class MyDataSet_Resample(Dataset):
    def __init__(self, enc_inputs, dec_inputs,txt_attn):
        super(MyDataSet_Resample, self).__init__()
        self.image = enc_inputs
        self.text = dec_inputs
        self.txt_attn = txt_attn

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        return self.image[idx], self.text[idx],self.txt_attn[idx]