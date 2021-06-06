FROM tensorflow/tensorflow:latest-gpu

ARG USR=kenders
ARG UID=1000
ARG DGID=""
ARG GID=1000

# User account creation that allows host Developer modifications.
# Create group with the GID of the host user.
# Create user with the UID of the host user.

RUN apt-get update
RUN apt-get install zsh git -y
RUN apt-get install vim -y

RUN apt-get install libsndfile1-dev libasound-dev portaudio19-dev \
    wget \
	libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    python3-venv \
    sox -y

RUN apt-get install sudo
# Create data group if specified as build argument, mirroring kronos' structure.
RUN if [ -n "${DGID}" ]; then groupadd -g $DGID -o data; fi
# Create the user and add them to their user group.
RUN groupadd -g $GID -o $USR \
    && useradd -m -u $UID -g $GID -o -s /bin/zsh $USR
# Add the user also to the data group if specified as build argument to be able to access /shared/data.
RUN if [ -n "${DGID}" ]; then usermod -a -G data $USR; fi

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


USER $USR
WORKDIR /home/${USR}
RUN mkdir /home/${USR}/.virtualenvs \
    && mkdir /home/${USR}/Developer
RUN pip install virtualenvwrapper \
    && rm -rf ~/.cache/pip

# Zsh setup.
RUN curl -L http://install.ohmyz.sh > install.sh && sh install.sh && rm install.sh .bash_logout .bashrc .profile \
    && echo "export WORKON_HOME=$HOME/.virtualenvs" >> /home/${USR}/.zshrc \
    && echo "export PROJECT_HOME=$HOME/Developer" >> /home/${USR}/.zshrc \
    && echo "source /usr/local/bin/virtualenvwrapper.sh" >> /home/${USR}/.zshrc
# Setup user python envs.
ENV VIRTUAL_ENV=/home/${USR}/.virtualenvs/main
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m virtualenv --python=/usr/bin/python3 --system-site-packages $VIRTUAL_ENV

# Main venv packages.
RUN pip install -U pip

#WORKDIR /clarity_CEC1/install


##########################################################################################################
# install clarity package
#RUN git clone https://github.com/kenders2000/clarity_CEC1
RUN git clone https://github.com/claritychallenge/clarity_CEC1.git
#Set working directory
WORKDIR /home/${USR}/clarity_CEC1/install
# from ./install_prerequisites.unbuntu.sh
RUN ./install.sh
WORKDIR /home/${USR}/clarity_CEC1/tools/openMHA
RUN ./configure && make
RUN make install
RUN source bin/thismha.sh
# RUN source /home/${USR}/clarity_CEC1/env/bin/activate
# install in the generic virtual env, not in the clarity one
WORKDIR /home/${USR}/clarity_CEC1/
RUN pip install -r requirements.txt



RUN pip install ipdb pudb jupyterlab matplotlib dash jupyter-dash plotly chart_studio librosa
# RUN pip install torch torchaudio
RUN pip install tensorflow_io tqdm tensorflow
RUN pip install jedi==0.17.2

# make sure that jupyter uses the venv
RUN pip install ipykernel
RUN python -m ipykernel install --user --name env --display-name "Clarity venv"
# setup a environment variable to points to clarity root
ENV CLARITY_ROOT="/home/${USR}/clarity_CEC1"
# directory to open when entering container
# WORKDIR /home/${USR}/greenhdd
# Build:
# docker build --build-arg USR=kenders --build-arg UID=$(id -u)  --build-arg GID=$(id -g) -f clarity.dockerfile -t kenders:tf-clarity .
#
# Run with:
# docker run --volume=/Users/kenders/Developer:/home/kenders/Developer -p 8888:8888 -it kenders:tf-clarity zsh

# to activate the virtualenv :
# source ${CLARITY_ROOT}/env/bin/activate
# then run the following to create the sym links to the data:
# (cd /home/kenders/greenhdd/clarity_challenge/pk_speech_enhancement && ./make_links.sh)
