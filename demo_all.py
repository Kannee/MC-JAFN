import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import torch.utils.data as data
from torch.autograd import Variable
import scipy.io
import h5py


# sensor = "QB/"
# model_path = 'model/model_for_QB.pth'
# save_path = 'result/QB_result.mat'
# dataset_path = 'data/QB_simulation_test.mat'
sensor = "WV4/"
model_path = 'model/model_for_WV4.pth'
save_path = 'result/WV4_result.mat'
dataset_path = 'data/WV4_simulation_test.mat'
Reduced_reslution_flag = 1
batchSize = 1


def main():
    print("===> Loading datasets")

    dataset = DatasetFromHdf5(dataset_path)
    data_loader = data.DataLoader(dataset=dataset, num_workers=0, batch_size=batchSize,
                                  shuffle=False, drop_last=False)

    model = torch.load(model_path)["model"]
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(data_loader, 1):

            ms, pan, gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

            ms = ms.cuda()
            pan = pan.cuda()
            gt = gt.cuda()

            out = model(ms, pan)
            loss = F.l1_loss(out, gt)
            print("RMSE_predicted=", loss)
            if iteration == 1:
                pred_ms = out
            else:
                pred_ms = torch.cat([pred_ms, out], 0)
            torch.cuda.empty_cache()
    gt = torch.from_numpy(dataset.gt)
    pred_ms = pred_ms.cpu()
    loss = F.l1_loss(pred_ms, gt)
    print("Total_RMSE_predicted=", loss)
    scipy.io.savemat(save_path, {'result': np.transpose(pred_ms.numpy().astype(np.float32), [0, 2, 3, 1])})


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path, 'r')
        ms = np.transpose(hf.get("ms"), [3, 2, 1, 0])  ##h5py读入数据格式
        pan = np.transpose(hf.get("pan"), [2, 1, 0])
        gt = np.transpose(hf.get("gt"), [3, 2, 1, 0])

        # hf = io.loadmat(file_path)
        # ms = hf['ms']
        # pan = hf['pan']
        # gt = hf['gt']

        self.ms = np.transpose(ms, [0, 3, 1, 2])  ##N*C*H*W
        self.gt = np.transpose(gt, [0, 3, 1, 2])
        self.pan = pan[:, np.newaxis, :, :]

    def __getitem__(self, index):
        return torch.from_numpy(self.ms[index, :, :, :]).float(), torch.from_numpy(
            self.pan[index, :, :, :]).float(), torch.from_numpy(self.gt[index, :, :, :]).float()

    def __len__(self):
        return self.ms.shape[0]


class DatasetFromHdf5Nogt(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5Nogt, self).__init__()
        hf = h5py.File(file_path, 'r')
        ms = np.transpose(hf.get("ms"), [3, 2, 1, 0])  ##h5py读入数据格式
        pan = np.transpose(hf.get("pan"), [2, 1, 0])

        self.ms = np.transpose(ms, [0, 3, 1, 2])  ##N*C*H*W
        self.pan = pan[:, np.newaxis, :, :]

    def __getitem__(self, index):
        return torch.from_numpy(self.ms[index, :, :, :]).float(), \
               torch.from_numpy(self.pan[index, :, :, :]).float()

    def __len__(self):
        return self.ms.shape[0]


if __name__ == "__main__":
    main()
