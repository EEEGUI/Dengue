import torch
import torch.utils.data
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
from utils import load_data_by_city, generate_sequence_feature, generate_submission_2
from sklearn.model_selection import train_test_split
from logger import Logger

BATCH_SIZE = 16
SEQUENCE_LENGTH = 10
INPUT_SIZE = 22
EVAL_SIZE = 0.3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_EPOCHS = 512
LEARNING_RATE = 0.01
LR_STEP_SIZE = NUM_EPOCHS // 3
LR_GAMMA = 0.1
DEVICE = torch.device('cuda')

# torch.backends.cudnn.enabled = True


class DengueData(torch.utils.data.Dataset):
    def __init__(self, city, dataset_name):
        """
        :param city:
        :param dataset_name: 数据集的哪一部分 train：训练集， val：验证集， test：测试集， train_all:训练集+验证集
        """
        self.city = city
        self.dataset_name = dataset_name
        data = load_data_by_city(city)
        train_features = generate_sequence_feature(data['train_feature'], SEQUENCE_LENGTH)
        train_label = data['train_label']
        test_feature = generate_sequence_feature(data['test_feature'], SEQUENCE_LENGTH)

        train_x, val_x, train_y, val_y = train_test_split(train_features,
                                                          train_label,
                                                          test_size=EVAL_SIZE,
                                                          shuffle=False)

        if self.dataset_name == 'train_all':
            self.features = train_features
            self.labels = train_label

        elif self.dataset_name == 'test':
            self.features = test_feature
            self.labels = np.zeros(len(test_feature))

        elif self.dataset_name == 'train':
            self.features = train_x
            self.labels = train_y

        elif self.dataset_name == 'val':
            self.features = val_x
            self.labels = val_y

        else:
            raise ValueError('Please input correct dataset name \n'
                             ' train：训练集， val：验证集， test：测试集， train_all:训练集+验证集')

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        feature = torch.from_numpy(feature)
        label = torch.from_numpy(np.array(label))
        return feature, label

    def __len__(self):
        return len(self.features)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        out, _ = self.lstm(x, (h0, c0)) # out shape=> (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train(city):
    print('start training ... ')
    is_val = False
    model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_data = DengueData(city, 'train_all')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    total_step = len(train_loader)

    logger = Logger('../output/logs_%s' % city)
    scheduler = StepLR(optimizer, LR_STEP_SIZE * total_step, LR_GAMMA)
    for epoch in range(NUM_EPOCHS):
        for i, (features, labels) in enumerate(train_loader):
            features = features.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE).float().to(DEVICE)
            labels = labels.to(DEVICE).float()
            outputs = model(features)
            print(outputs)
            loss = criterion(outputs.reshape(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i+1) % 5 == 0:
                print('Epoch[%d/%d], step[%d/%d], Loss:%.5f' % (epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))
                # print('epoch:%d, step:%d, learning_rate:%.5f' % (epoch, i, scheduler.get_lr()[0]))
                logger.scalar_summary('loss', loss.item(), epoch * total_step + i + 1)
                logger.scalar_summary('learning_rate', scheduler.get_lr()[0], epoch * total_step + i + 1)

        if is_val:
            print('Epoch % d Start validating' % epoch)
            val_data = DengueData(city, 'val')
            val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=16, shuffle=False)
            with torch.no_grad():
                mae = 0
                val_size = len(val_data)
                for features, labels in val_loader:
                    features = features.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE).float().to(DEVICE)
                    labels = labels.to(DEVICE).float()
                    outputs = model(features)
                    mae += criterion(outputs.reshape(-1), labels) * labels.size(0) / val_size
                print('The MAE of val dataset is %.5f' % mae)
                logger.scalar_summary('accuracy', mae, epoch)

    torch.save(model.state_dict(), '../config/%s_model.ckpt' % city)


def predict(city):
    model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load('../config/%s_model.ckpt' % city))
    print('Start predicting')
    test_data = DengueData(city, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1000, shuffle=False)
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE).float().to(DEVICE)
            outputs = model(features)
    outputs = outputs.reshape(-1).cpu().numpy()
    logger = Logger('../output/logs_%s' % city)
    for i, each in enumerate(outputs):
        logger.scalar_summary('prediction', int(each), i+1)
    return outputs


if __name__ == '__main__':
    train('sj')
    train('iq')
    sj = predict('sj')
    iq = predict('iq')
    generate_submission_2(sj, iq)








