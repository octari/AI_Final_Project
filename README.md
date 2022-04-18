# Foundations of AI Project: Voice Liveness Detection Systems: PhonemeLive & VoidNet
## About/Overview
Voice liveness detection is a comprehensive multidisciplinary field and a wide-ranging interdisciplinary subject. It also has very close links with many disciplines such as phonetics, linguistics, acoustics, cognitive science, physiology, psychology, etc.
We humans have increasingly started using voice as a User Interaction (UI) channel. However the open nature of the voice channel makes it vulnerable to various attacks. One of the most basic and widely used types of attack is the replay attack, where the attacker uses a speaker to replay a previously recorded live human voice.

To mitigate this type of attack, we present two voice liveness detection systems:
- PhonemeLive
- VoidNet

## List of Approaches
#### PhonemeLive
- An audio classification system that performs phoneme segmentation and extracts light-weight phoneme-based features with a neural network as classifier.

#### VoidNet
- An audio classification system that uses frequency-dependent spectral power features from Void along with embeddings from the intermediate layer of a neural network trained on the spectrogram of the input wave.

## Dataset
- ASVSpoof Dataset for PhonemeLive and VoidNet: https://datashare.ed.ac.uk/handle/10283/3055

## How to Run
- Please head to Installation.txt files present in PhenomeLive and VoidNet to run the programs for each type respectively.

## Citations
- Yan, Q., Liu, K., Zhou, Q., Guo, H., and Zhang, N. Surfingattack: Interactive hidden attack on voice assistants using ultrasonic guided waves. In Network and Distributed Systems Security (NDSS) Symposium (2020).
- Maeda, S. (1982). A digital simulation method of the vocal-tract system. Speech Communication, 1(3-4), 199–229. https://doi.org/10.1016/0167-6393(82)90017-6
- T. Ganchev, N. Fakotakis, and G. Kokkinakis (2005), "Comparative evaluation of various MFCC implementations on the speaker verification task Archived 2011-07-17 at the Wayback Machine," in 10th International Conference on Speech and Computer (SPECOM 2005), Vol. 1, pp. 191–194.
- Tom, F., Jain, M., and Dey, P. End-to-end audio replay attack detection using deep convolutional networks with attention. In Interspeech 2018, 19th Annual Conference of the International Speech Communication Association, Hyderabad, India, 2-6 September 2018 (2018),
B. Yegnanarayana, Ed., ISCA, pp. 681–685.