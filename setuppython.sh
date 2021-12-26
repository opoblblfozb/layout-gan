# optuna用 postgres client
sudo apt update
sudo apt install postgresql postgresql-contrib

# python3.9環境の用意
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get -y update
sudo apt-get -y install python3.9
sudo apt-get -y install python3.9-dev
sudo apt-get -y install python3-pip
sudo apt-get -y install python3.9-distutils
python3.9 -m pip install --upgrade setuptools
python3.9 -m pip install --upgrade pip
python3.9 -m pip install --upgrade distlib

# pythonでpython3.9.7を呼べるように
echo export PATH='/usr/bin:$PATH' >> ~/.bashrc
sudo ln -sf /usr/bin/python3.9 /usr/bin/python
source ~/.bashrc


# python packageの準備
pip install poetry