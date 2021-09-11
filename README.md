# Clarity Enhancement challenge

This my entry to the [clarity enhancement challenge](https://github.com/claritychallenge/clarity_CEC1/) to enhance speech for the hearing impaired.

# Installation
1) install docker
  https://docs.docker.com/engine/install/ubuntu/
2) clone this repo
  ```bash:
  git clone https://github.com/kenders2000/u_net_speech_enhancement.git
  ```
3) Build the container image, add your username to the build command. This will
  pass through the permissions required to read and write as that user within
  the container. This will pull the latest tensorflow-gpu image and build upon
  that. This will work for both gpu and non gpu machines. Note this will pull the
  latest clarity  challenge repo into the image.
  ```bash:
  docker build --build-arg USR=username --build-arg UID=$(id -u)  --build-arg GID=$(id -g) -f clarity.dockerfile -t username:tf-clarity .
  ```

4) run the container, attach the u_net_speech_enhancement folder.
  ```bash:
  docker run --volume=/path/to/u_net_speech_enhancement:/home/ubuntu/u_net_speech_enhancement --volume=/path/to/data:/home/ubuntu/data -p 8888:8888 -it username:tf-clarity zsh
  ```

5) additional environment steps:

  Setup all the symlinks,  in the following we point to the data in
  `/home/username/u_net_speech_enhancement/example_data` this contains a few examples
  from the dev set of the clarity challenge, if you have the full data set, replace
  this with the path to parent folder of the `clarity_CEC1_data` folder.
  ```bash:
  (cd /home/username/u_net_speech_enhancement/env && ./make_links.sh username /home/username/u_net_speech_enhancement/example_data)
  ```

  Set up the openMHA binaries to be easily executable.
  ```bash:
  source /home/kenders/clarity_CEC1/tools/openMHA/bin/thismha.sh
  ```

  Note: I do not use the Clarity virutal env, but to activate that if required:
  ```bash:
  source ${CLARITY_ROOT}/env/bin/activate
  ```



# To train a model.
`python train_unet.py -c /path/to/model_checkpoints`

# To Predict the cleaned audio.
`python predict_with_trained_unet.py -p /path/to/trained_model -d <dataset>`

Saves into the clarity repo via the symlink, into the appropriate dataset dir.

to predict using the example dataset:

`python predict_with_trained_unet.py -p /path/to/trained_model -d dev`

# To generate evaluation data.
`python post_process_cleaned_audio_eval.py -p /path/to/trained_model -d <dataset> -i /path/to/cleaned_data -o /path/to/write_hearing_aid_files -s /path/to/clarity/dataset`

# To get SII for dev data:
post_process_cleaned_audio_with_mbstoi.py
