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

# gcc

 apt install gcc-11 g++-11

# pg install

apt-get update && apt-get install apt-utils && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
 #
 ## install pg_config
apt-get update && apt-get install wget ca-certificates -y && apt-get install -y gnupg2
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt bullseye-pgdg main" > /etc/apt/sources.list.d/pgdg.list'&& apt-get update && apt-get install postgresql postgresql-contrib -y

apt update && apt install git -y && apt install unzip -y && apt install docker-compose -y && apt install postgresql -y

git clone https://github.com/daviddhc20120601/chat-with-pdf.git && cd chat-with-pdf/

git checkout llama2

#cp .devops/Dockerfile . && docker build . -t haidonggpt/front:1.0   && docker run -d -e /etc/environmentadb -p 8501:8501 haidonggpt/front:1.0  \n



