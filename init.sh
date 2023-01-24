#!/bin/bash

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
sudo python3 -m pip install pandas