# speech-scorer

Python Version – 3.7.1

## Development Guidelines

- Install repo
- Set up virtual env
- Install dependencies with `pip install -r requirements.txt`
- Create `data` folder under root of project and place train, dev and test data there
- Create `embeddings` folder in root of project
- Place glove embeddings in `embeddings` folder
- Create `checkpoints` folder in root of project (models and weights will be saved here)
- To run a model, run `python cnn_lstm_att.py --prompt_id <id>`
