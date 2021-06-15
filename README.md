# Installation
1) install docker
  https://docs.docker.com/engine/install/ubuntu/
2) clone this repo
  ```bash:
  git clone https://github.com/kenders2000/spectrogram_based_model.git
  ```
3) Build the container image, add your username to the build command. This will
  pass through the permissions required to read and write as that user within
  the container. This will pull the latest tensorflow-gpu image and build upon
  that. This will work for both gpu and non gpu machines.
  ```bash:
  docker build --build-arg USR=username --build-arg UID=$(id -u)  --build-arg GID=$(id -g) -f clarity.dockerfile -t username:tf-clarity .
  ```
4) run the container, attach the spectrum_based_model folder.
  ```bash:
  docker run --volume=/home/ubuntu/spectrogram_based_model:/home/ubuntu/spectrogram_based_model -p 8888:8888 -it username:tf-clarity zsh
  ```
5) additional environment steps:
```bash:
source ${CLARITY_ROOT}/env/bin/activate
(cd /home/ubuntu/spectrogram_based_model/env && ./make_links.sh ubuntu /home/ubuntu/spectrogram_based_model/example_data)
source /home/ubuntu/clarity_CEC1/tools/openMHA/bin/thismha.sh
```

# To train a model.
`python train_unet.py -c /path/to/model_checkpoints`

# To Predict the cleaned audio.
`python predict_with_trained_unet.py -p /path/to/trained_model -d <dataset>`

Saves into the clarity repo via the symlink, into the appropriate dataset dir.

# To generate evaluation data.
`python post_process_cleaned_audio_eval.py -p /path/to/trained_model -d <dataset> -i /path/to/cleaned_data -o /path/to/write_hearing_aid_files -s /path/to/clarity/dataset`

# To get SII for dev data:
post_process_cleaned_audio_with_mbstoi.py
