# aai3008-group-8

## Contributions
- MUHAMMAD FIRDAUZ BIN KAMARULZAMAN
- SHERWYN CHAN YIN KIT
- MUHAMMAD AKID NUFAIRI BIN NASHILY
- DOMINICK SEAH ZI YU
- QUAY JUN WEI

## Getting started

### We'll be using venv as our virtual environment
```
pip install virtualenv
```

1. **Create virtual environment**

Mac
```
python3 -m venv myenv
```

Windows
```
python -m venv myenv
```

2. Activate virtual environment

Mac
```
source myenv/bin/activate
```

Windows
```
source myenv/Scripts/activate
```


3. Install required libraries & dependancies 
```
pip install -r requirements.txt
```

## Project Organisation
```
├── src/                       # Source code
│   ├── asr/                   # Automatic Speech Recognition module
│   ├── llm/                   # Language Learning Model module
│   ├── transcript/            # Transcript processing module
│   └── main.py                # Streamlit app
├── .gitignore                 # Git ignore file
├── README.md                  # Readme file
└── requirements.txt           # Project dependencies
```