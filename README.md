# OLKAVS: An Open Large-Scale Korean Audio-Visual Speech Dataset
---

This repository contains code scripts for training and evaluation of the OLKAVS dataset described in the paper [OLKAVS: An Open Large-Scale Korean Audio-Visual Speech Dataset](www.github.com).


  * [Datasets](#datasets)
  * [Inference](#evaluation)
  * [Contacts](#contacts)


## Datasets

|<img src="./assets/sample_0.gif" width="512px" title="Sample_0"/>|<img src="./assets/sample_1.gif" width="512px" title="Sample_1"/>|
| :--: | :--: |
| **Sample #1 : {A,B,D,F,H}**<br>"그때 되게 한여름이어서 되게 뜨거웠거든요." | **Sample #2 : {A,C,E,G,I}**<br>"그래서 도서관엘 다시 들어갔어요 공부하기 위해서" |
<!-- ![Sample_0](./assets/sample_0.gif) -->

The OLKAVS contains below.

- a total of 1,150 hours of audio
- a total of 5,750 hours of synced video from 9 different viewpoints

Those are from 1,107 Korean speakers in a studio setup with corresponding Korean transcriptions.

Yo can download The OLKAVS datasets from [AIHub: 립리딩(입모양) 음성인식 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=538).

### Dataset Structure

#### Directory
The folder structure of the OLKAVS dataset is as follows:
```
{root}/{group}/{subgroup}/{noise}/{specificity}/{gender_group}/{gender_subgroup}/{session_idx}.{extension}
```
- `root` : Root directory
- `group` : Data grouped by usage. (e.g. Train, Validation, Test)
- `subgroup` : Data are separated in subgroups randomly.
- `noise` : Noise condition.
- `specificity` : Specificity of speaker.
- `gender_group` : Gender.
- `gender_subgroup` : Data of each gender are separted in gender_subgroups randomly.
- `session_idx` : Index of the 5-minutes-long recording session. This index follows the name rule at [here](#name-rule).
- `extension` : File extension. `mp4`, `wav`, `json` for video, audio and label, respectively.

example
```
./원천데이터/TS1/소음환경1/C(일반인)/F(여성)/F(여성)_1/lip_J_1_F_02_C032_A_010.wav
```

##### Name Rule
The rule of naming file is as follows:
```
lip_{video_env}_{audio_noise}_{gender}_{age}_{specificity}{speakerID}_{video_angle}_{index}
```
- `video_env`
J: indoor, K:outdoor
- `audio_noise`
1: No noise, 2: Indoor noise, 3: Indoor ambiance, 4: Traffic noise, 5:Construction-site noise, 6: Natural outdoor noise
- `gender`
F : Female, M : Male
- `age`
1 : 10 - 19, 2 : 20 - 29, 3: 30 - 39, 4: 40 - 49, 5: 50 - 59, 6: 60 over
- `specificity`
C : Common speaker , E: Expert
- `speakerID`
Identified number for speaker
- `video_angle`
A : Frontal, B : Upper left, C : Left, D : Lower left, E : Lower center, F : Lower right, G : Right, H : Upper right, I : Upper center
- `index`
Index of the 5-minute-long recording session

#### Label Structure (json)

```
├── dataSet
│   ├── description
│   ├── url
│   ├── version
│   └── year
│
├── Video_info
│   ├── video_Name
│   ├── video_Format
│   ├── video_Duration
│   ├── FPS
│   └── Resolution
│
├── Audio_info
│   ├── Audio_Name
│   ├── Audio_Format
│   ├── Audio_Duration
│   ├── Sampling_rate
│   └── Channel(s)
│
├── Audio_env
│   └── Noise
│
├── Video_env
│   ├── env
│   └── Angle
│
├── Sentence_info
│   ├── ID
│   ├── topic
│   ├── sentence_text
│   ├── start_time
│   └── end_time
│
├── speaker_info
│   ├── speaker_ID
│   ├── Specificity
│   ├── Gender
│   ├── Age
│   └── Accent
│
└── Bounding_box_info
    ├── Face_bounding_box
    └── Lip_bounding_box
```

## Dependency

```bash
pip install -r requirements.txt
```

## Pre-process

Preprocess the data. 
Crop the lengths of audio and video by the temporal label. (start, end) \
Then crop the video to the shape (96 96), by bounding box. \
Finally generate label scripts for training or evaluation. 

**Preparation**

Data folder should comply with [this](#directory) structure

**Run Script**
```bash
python preprocess.py --root_dir {ROOT_DIR} --src_dir {SOURCE_DIR} --label_dir {LABEL_DIR}
```

**Generated Label Samples**
```
{Video_filepath}\t{Audio_filepath}\t{Transcription}\t{Tokenized_Numbers}
```
```
./save/원천데이터/TS1/소음환경1/C(일반인)/F(여성)/F(여성)_1/lip_J_1_F_02_C032_A_011/2.mp4	./save/원천데이터/TS1/소음환경1/C(일반인)/F(여성)/F(여성)_1/lip_J_1_F_02_C032_A_011/2.wav	건강이 안 좋아지니 죄착 죄책감이 드네	5 28 48 5 24 65 16 44 4 16 24 48 4 17 32 71 16 24 17 44 7 44 4 17 35 19 24 45 4 17 35 19 25 45 5 24 60 16 44 4 8 42 7 29
./save/원천데이터/TS1/소음환경1/C(일반인)/F(여성)/F(여성)_1/lip_J_1_F_02_C032_A_011/3.mp4	./save/원천데이터/TS1/소음환경1/C(일반인)/F(여성)/F(여성)_1/lip_J_1_F_02_C032_A_011/3.wav	요즘에 불면증이 심해진 것 같아	16 36 17 42 60 16 29 4 12 37 52 11 30 48 17 42 65 16 44 4 14 44 60 23 25 17 44 48 4 5 28 63 4 5 24 69 16 24
...
```

## Extract Lip Feature (Optional)
To reduce the required memory resource, this script extracts lip features in advance.

```
Being in the works...
```

## Evaluation
### Inference
**Run**
```bash
python inference.py -c {CONFIG_FILE_PATH}
```

### Results
| Model         | # of params |  TrainData | CER | WER | sWER | pt |
| ------------- | :---------: | :--------: | :-: | :-: | :--: | -- |
| `AV-model`    |     62M*    |    None    |  5.64 | 12.05 |  9.45 |[here](https://drive.google.com/drive/folders/1sElJn4efJdMRabMqk7-p6L7_3Dii5bW8?usp=share_link)|
| `A-model`     |     38M*    |    None    |  5.63 | 11.61 |  9.37 |                     |
| `V-model`     |     34M*    |    None    | 36.53 | 49.97 | 51.71 |                     |
| `F-model`     |     45M     |    None    | 55.00 | -     |     - |                     |
| `All-model`   |     45M     |    None    | 44.86 | -     |     - |[here](https://drive.google.com/drive/folders/1sElJn4efJdMRabMqk7-p6L7_3Dii5bW8?usp=share_link)|

(* Do not include [pre-trained visual front-end](#extract-lip-feature-optional) parameters.)


## Release

* v1.0.0
  * baseline released

## License

The dataset itself is released under custom [terms and conditions](https://aihub.or.kr/intrcn/guid/usagepolicy.do).

The OLKAVS Scripts are released under MIT license.

## Citation

```
@misc{park2023olkavs,
    title={OLKAVS: An Open Large-Scale Korean Audio-Visual Speech Dataset},
    author={Jeongkyun Park and Jung-Wook Hwang and Kwanghee Choi and Seung-Hyun Lee and Jun Hwan Ahn and Rae-Hong Park and Hyung-Min Park},
    year={2023},
    eprint={2301.06375},
    archivePrefix={arXiv},
    primaryClass={cs.MM}
}
```

## Contacts

`park32323@gmail.com` [@Park323](https://github.com/Park323)
