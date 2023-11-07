import torch
import torch.nn as nn

class sampleSpecAE(nn.Module):

    '''
    A sample Spectrogram-based Autoencoder, which reconstructs the audio as it is!
    Spectrogram setting: n_fft=256, hop_length = 128, power=None
    '''

    def __init__(self):
        super(sampleSpecAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU()
            )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(2,3), stride=(2), padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=(2,2), stride=(2), padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=(3), stride=(1), padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=(2,3), stride=(2), padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 512, kernel_size=(3,3), stride=(2), padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)

            return x
        
