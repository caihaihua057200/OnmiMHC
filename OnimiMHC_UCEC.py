import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
allele_list = [
    "A1101",
    "A0201",
    "A2402",
    "A0301",
    "A3303",
    "B1501",
    "B4001",
    "B4601",
    "B5801",
    "B0702",
    "B1301",
    "B0801",
    "B3501",
    "B5701",
    "C0304",
    "C0702",
    "C0102",
    "C0401",
    "C0602",
    "C0801"
]

BLO = pd.read_csv(fr'./blosum62.csv', header=None)
input_file  = sys.argv[1]
output_file = sys.argv[2]
TEST = pd.read_csv(fr'{input_file}')
PSEDUO = pd.read_csv(r'./pseudo.csv')
peptide = TEST['Peptide'].str.slice(start=-11).str.upper().str.pad(width=11, side='right', fillchar='-')

amino_acids = "ACDEFGHIKLMNPQRSTVWYXU-"
P_SEQ = [list(p) for p in peptide]
P_encoder = OneHotEncoder(
    categories=[list(amino_acids)] * 11)
P_encoder.fit(P_SEQ)
P_SEQ = P_encoder.transform(P_SEQ).toarray()

BLO_dict = {x: np.array(BLO.loc[BLO[0] == x, 1:].values[0]) for x in BLO[0]}
B = np.array([[BLO_dict[j] for j in i] for i in peptide], dtype='float32')

P_SEQ = torch.tensor(P_SEQ, dtype=torch.float32)
B = torch.tensor(B)
B = B.permute(0, 2, 1)

for z in allele_list:
    A = PSEDUO[PSEDUO['H'] == z]['P'].values
    print(z)
    TEST['HLA_sequence'] = A[0]
    allele = TEST['HLA_sequence'].str.pad(width=34, side='right', fillchar='-')
    SEQ = peptide.str.cat(allele, sep='')
    SEQ = [list(p) for p in SEQ]
    encoder = OneHotEncoder(
        categories=[list(amino_acids)] * 45)
    encoder.fit(SEQ)
    SEQ = encoder.transform(SEQ).toarray()
    ########################################################################################################################
    SEQ = torch.tensor(SEQ, dtype=torch.float32)
    dataset_train = TensorDataset(P_SEQ, SEQ, B)
    test_order = DataLoader(dataset_train, shuffle=False, batch_size=10000)
    P =[]
    for zzz in tqdm(range(5)):
        model_path = fr'./weights/model_{zzz}.pt'
        class CBAM(nn.Module):
            def __init__(self, channel, reduction=16, spatial_kernel=7):
                super(CBAM, self).__init__()
                self.max_pool = nn.AdaptiveMaxPool2d(1)
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.mlp = nn.Sequential(
                    nn.Conv2d(channel, channel // reduction, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // reduction, channel, 1, bias=False)
                )
                self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                      padding=spatial_kernel // 2, bias=False)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                max_out = self.mlp(self.max_pool(x))
                avg_out = self.mlp(self.avg_pool(x))
                channel_out = self.sigmoid(max_out + avg_out)
                x = channel_out * x
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                avg_out = torch.mean(x, dim=1, keepdim=True)
                spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
                x = spatial_out * x
                return x
        class model1_5(nn.Module):
            def __init__(self,
                         esm_embedding_size=1280 * 11,
                         fc1_size=1800,
                         fc2_size=1000,
                         fc3_size=450,
                         fc1_dropout=0.3,
                         fc2_dropout=0.3,
                         fc3_dropout=0.3,
                         num_of_classes=2):
                super(model1_5, self).__init__()

                self.f_model = nn.Sequential(nn.Linear(8472, fc1_size),  # 4624   4424+2688+704
                                             nn.BatchNorm1d(fc1_size),
                                             nn.ReLU(),
                                             nn.Dropout(fc1_dropout),
                                             nn.Linear(fc1_size, fc2_size),
                                             nn.BatchNorm1d(fc2_size),
                                             nn.ReLU(),
                                             nn.Dropout(fc2_dropout),
                                             nn.Linear(fc2_size, fc3_size),
                                             nn.BatchNorm1d(fc3_size),
                                             nn.ReLU(),
                                             nn.Dropout(fc3_dropout),
                                             nn.Linear(fc3_size, num_of_classes),
                                             # nn.Softmax(dim=1)
                                             #    nn.Sigmoid()
                                             )

                self.conv_layers = nn.Sequential(
                    nn.Conv1d(23, 64, kernel_size=1),
                    nn.BatchNorm1d(64),
                    nn.Dropout(fc3_dropout),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                )

                hidden_dim = 64
                self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True,
                                    # dropout=fc3_dropout,
                                    bidirectional=True)

                self.conv_2D = nn.Sequential(
                    nn.Conv2d(1, 16, 1),
                    nn.BatchNorm2d(16),
                    nn.Dropout(fc3_dropout),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
                self.image_cbam = CBAM(16)

            def forward(self, x, z, b):
                x = x.view(x.shape[0], 11, -1)  # P_SEQ
                z = z.view(x.shape[0], 45, -1)


                zz = self.conv_2D(z.unsqueeze(1))
                az = self.image_cbam(zz)
                zz = zz * az
                zz = torch.reshape(zz, (zz.shape[0], zz.shape[1] * zz.shape[2] * zz.shape[3]))

                zzz = z.permute(0, 2, 1)
                zzz = self.conv_layers(zzz)
                zzz = zzz.transpose(1, 2)
                zzz, _ = self.lstm(zzz)
                zzz = torch.reshape(zzz, (zzz.shape[0], zzz.shape[1] * zzz.shape[2]))

                xx = x.unsqueeze(1)
                xx = self.conv_2D(xx)
                a = self.image_cbam(xx)
                xx = a * xx
                xx = torch.reshape(xx, (xx.shape[0], xx.shape[1] * xx.shape[2] * xx.shape[3]))

                x = x.permute(0, 2, 1)
                x = self.conv_layers(x)
                x = x.transpose(1, 2)
                x, _ = self.lstm(x)

                x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
                b = torch.reshape(b, (b.shape[0], b.shape[1] * b.shape[2]))
                IN = torch.cat((zz, zzz, x, xx, b), dim=1)
                out = self.f_model(IN)
                return out
        model = model1_5()
        softmax_function = nn.Softmax(dim=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        PREDICT = []
        with torch.no_grad():
            for p, s, b in tqdm(test_order):
                output = model(p.to(device), s.to(device),b.to(device))
                output1 = softmax_function(output)[:, 1]
                PREDICT.extend(output1.cpu().numpy())
        P.append(PREDICT)
    P = np.mean(P, axis=0)
    TEST[f'{z}'] = P
TEST = TEST.drop('HLA_sequence', axis=1)
TEST.to_csv(fr'{output_file}',index=False)

