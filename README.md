 # Automatic Speech Emotion Recognition (SER) system

### Introduction
This project focuses on building an Automatic Speech Emotion Recognition (SER) system capable of detecting emotions from audio speech data. 
By leveraging deep learning models such as ResNet50, we aim to classify emotions like anger, disgust, fear, joy, neutral, sadness, and surprise from speech audio clips. 
The project uses advanced audio augmentation techniques and models trained on diverse datasets to ensure robust emotion detection across different speakers and scenarios.

### Requirement
The following Python packages are required for the project:

  * torchaudio
  * pytorch-ignite
  * wandb -qqq
  * audiomentations
  * pydub
  * boto3
  * wandb

### Dataset
The dataset used for this project is the UrbanSoundDataset, which contains various speech audio clips labeled with different emotions. 
The dataset is split into training and testing sets with multiple categories such as anger, disgust, fear, joy, neutral, sadness, and surprise.

Here is an example of how the dataset is structured:
| Filename | Emotion |
| ------------- | ------------- |
| 18777.mp3 |	Neutral |
| 24041.mp3 |	Neutral |
| 1621.mp3 |	Joy |
| 28883.mp3 |	Neutral |

The dataset is essential for training the model to understand the subtle nuances in human speech that convey different emotions.

### Preprocessing
To improve the model's performance and robustness, we apply various audio augmentation techniques. These include:

| Augmentation | Details |
| ------------- | ------------- |
| AddGaussianNoise | Adds random noise to make the model more robust to noisy environments. |
| TimeStretch | Stretches or compresses the time of the audio signal. |
| PitchShift | Changes the pitch of the audio to simulate different voices. |
| Shift | Shifts the audio to the left or right, adding temporal augmentation. |

Additional audio effects applied during preprocessing:

| Effects  | Details |
| ------------- | ------------- |
| lowpass | Apply single-pole lowpass filter |
| speed | Reduce the speed (changes sample rate) |
| rate | Add `rate` effect with original sample rate |
| reverb | Add reverberation to give a dramatic effect |

### Training and Evaluation
The training process utilizes a deep learning model like ResNet50, a robust convolutional neural network known for handling image and audio classification tasks effectively. To train the model:

Run the training script using the command:
```nohup python train.py &```

The model's performance is evaluated using standard metrics such as accuracy, precision, recall, and F1-score.

### References
  * [A Comprehensive Review of Speech Emotion Recognition Systems](https://ieeexplore.ieee.org/document/9383000)
  * [An ongoing review of speech emotion recognition](https://www.sciencedirect.com/science/article/pii/S0925231223000103)
  * [Speech emotion recognition using machine learning — A systematic review](https://www.sciencedirect.com/science/article/pii/S2667305323000911)
  * [Foundation Model Assisted Automatic Speech Emotion Recognition: Transcribing, Annotating, and Augmenting](https://arxiv.org/abs/2309.08108)
  * [Speech Emotion Recognition Systems: A Comprehensive Review on Different Methodologies](https://dl.acm.org/doi/abs/10.1007/s11277-023-10296-5)
