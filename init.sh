#!/bin/bash

# Excutor Setup

sudo nano /etc/hosts #name the ip adress
<<com
172.31.31.202 CentralManager
172.31.24.64 Submission
com

## EXECUTE ##
curl -fsSL https://get.htcondor.org | sudo /bin/bash -s -- --no-dry-run --password "PASSWORD_HERE" --execute CentralManager


# creating dir
sudo mkdir /home/ubuntu/efs

# installing efs-utils
sudo apt-get update
sudo apt-get -y install git binutils
git clone https://github.com/aws/efs-utils
cd /home/ubuntu/efs-utils
./build-deb.sh
sudo apt-get -y install ./build/amazon-efs-utils*deb


# mounting efs
sudo mount -t efs -o tls fs-00b6383753ff70712:/ /home/ubuntu/efs

# instaling package
sudo apt-get -y install python3-pip

# fetting requiemtns 
wget https://raw.githubusercontent.com/Sarmin-smrity/aws-project/main/requirements.txt

# isntall requiremnts
#pip install -r requirements.txt

pip3 install -U scikit-learn numpy==1.24.1 matplotlib==3.6.3 seaborn pandas
pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html


# download data
wget https://github.com/Sarmin-smrity/aws-project/raw/main/data.zip

#download sample code
wget https://raw.githubusercontent.com/Sarmin-smrity/aws-project/main/code/resnet18.py
