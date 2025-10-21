# lstm.py
import torch
import torch.nn as nn

class RowLSTM(nn.Module):

    def __init__(self, input_size=28, hidden_size=128, num_layers=1,
                 num_classes=10, dropout_p=0.2, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        d = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * d, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        x = x.squeeze(1)  # -> [batch_size, 28, 28]
        out, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            h = torch.cat((hn[-2], hn[-1]), dim=1)  # concatenate both directions
        else:
            h = hn[-1]
        return self.fc(h)
