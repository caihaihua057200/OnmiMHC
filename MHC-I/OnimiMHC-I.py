import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BLO = pd.read_csv(fr'./blosum62.csv', header=None)
PSEDUO = pd.read_csv(fr'./pseudo.csv')
print(PSEDUO)
input_file  = sys.argv[1]
output_file = sys.argv[2]
TEST = pd.read_csv(fr'{input_file}')
F = TEST.copy()
merged = F.merge(PSEDUO, how='left', left_on='Allele', right_on='H')
F['Allele'] = merged['P'].combine_first(F['Allele'])
peptide_TEST = F['Peptide']
allele_TEST = F['Allele']
print(allele_TEST)


peptide = peptide_TEST.str.slice(start=-11).str.upper().str.pad(width=11, side='right', fillchar='-')
allele = allele_TEST.str.pad(width=34, side='right', fillchar='-')
SEQ = peptide.str.cat(allele, sep='')
BLO_dict = {x: np.array(BLO.loc[BLO[0] == x, 1:].values[0]) for x in BLO[0]}
B = np.array([[BLO_dict[j] for j in i] for i in peptide], dtype='float32')
amino_acids = "ACDEFGHIKLMNPQRSTVWYXU-"
P_SEQ = [list(p) for p in peptide]
P_encoder = OneHotEncoder(
    categories=[list(amino_acids)] * 11)
P_encoder.fit(P_SEQ)
P_SEQ = P_encoder.transform(P_SEQ).toarray()
SEQ = [list(p) for p in SEQ]
encoder = OneHotEncoder(
    categories=[list(amino_acids)] * 45)
encoder.fit(SEQ)
SEQ = encoder.transform(SEQ).toarray()
########################################################################################################################
P_SEQ = torch.tensor(P_SEQ, dtype=torch.float32)
SEQ = torch.tensor(SEQ, dtype=torch.float32)
B = torch.tensor(B)
B = B.permute(0, 2, 1)
test_order = TensorDataset(P_SEQ, SEQ, B)
########################################################################################################################
batch_size = 300
test_order = DataLoader(test_order, shuffle=False, batch_size=batch_size)
softmax_function = nn.Softmax(dim=1)
P =[]
for f in tqdm(range(5)):
    model_path = fr'./weights/model_{f}.pt'
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
    class model(nn.Module):
        def __init__(self,
                     fc1_size=1800,
                     fc2_size=1000,
                     fc3_size=450,
                     fc1_dropout=0.3,
                     fc2_dropout=0.3,
                     fc3_dropout=0.3,
                     num_of_classes=2):
            super(model, self).__init__()

            self.f_model = nn.Sequential(nn.Linear(8472, fc1_size),
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
    model = model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    PREDICT = []
    TRUE=[]
    with torch.no_grad():
        for P_SEQ, SEQ, B in tqdm(test_order):
            output = model(P_SEQ.to(device), SEQ.to(device), B.to(device))
            output1 = softmax_function(output)[:, 1]
            PREDICT.extend(output1.cpu().numpy())
    P.append(PREDICT)
P = np.mean(P, axis=0)
TEST['OmniMHC'] = P
print(TEST)
TEST.to_csv(fr'{output_file}')


