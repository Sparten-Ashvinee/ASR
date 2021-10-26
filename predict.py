import torch
import torchaudio
from sklearn.metrics import confusion_matrix
from resnet50_model import resnet
from process_data import get_data
from process_data import UrbanSoundDataset

#from cnn import CNNNetwork
#from urbansounddataset import UrbanSoundDataset
#from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

# ANNOTATIONS_FILE = '/content/drive/MyDrive/Cogito HE Challenge/dataset/test500.csv'
# AUDIO_DIR = '/content/drive/MyDrive/Cogito HE Challenge/dataset/train/'

def predict(model, input, target):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        #predicted = class_mapping[predicted_index]
        #expected = class_mapping[target]
    return predicted_index, target


def predictions(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, class_mapping, device):


    # load back the model
    #cnn = CNNNetwork()
    state_dict = torch.load("feedforwardnet.pth")
    cnn = resnet(device)
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    filenames, labels = get_data(ANNOTATIONS_FILE, AUDIO_DIR)

    usd = UrbanSoundDataset(filenames, labels,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "gpu:0")

    pre, exp = [], []
    for itm in usd:
        # get a sample from the urban sound dataset for inference
        input, target = itm[0], itm[1] # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)
        input = input.to(device)
        #print(input.type())
        #print(input.shape)
        #target = class_mapping.index(target)
        # make an inference
        predicted, expected = predict(cnn, input, target, class_mapping)
        pre.append(predicted)
        exp.append(expected)
    print(f"Predicted: '{predicted}', expected: '{expected}'")

    prr = []
    for it in pre:
      prr.append(it.item()) 
    
    cm = confusion_matrix(exp, prr)

    ccount = 0
    for i in range(len(prr)):
      if exp[i] == prr[i]:
        ccount+=1
    print('Test accuracy: ', (ccount/len(prr))*100)


