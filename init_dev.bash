#!/bin/bash\n

apt remove python3.10 -y

apt install software-properties-common -y
apt-get install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa && apt update

apt install python3.11 -y
apt install python3.11-dev -y
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
apt install python3-pip -y

apt install python3.11-distutils -y

apt update && apt install git -y && apt install unzip -y && apt install docker-compose -y && apt install postgresql -y

git clone https://github.com/daviddhc20120601/chat-with-pdf.git && cd chat-with-pdf/

git checkout llama2

#cp .devops/Dockerfile . && docker build . -t haidonggpt/front:1.0   && docker run -d -e /etc/environmentadb -p 8501:8501 haidonggpt/front:1.0  \n



