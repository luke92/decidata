# decidata
Machine Learning

# Comandos para preparar ambiente

# Instalar Python 3
sudo yum install python3

# Poner cómo default python 3 si ya está instalado python 2
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# Configurar yum con python 2.7
Change in top of file /bin/yum
#!/usr/bin/python
for
#!/usr/bin/python2.7

# Configurar urlgrabber-ext-down  con python 2.7
Change in top of file /usr/libexec/urlgrabber-ext-down
#!/usr/bin/python
for
#!/usr/bin/python2.7

# Instalar PIP
sudo yum install python-pip

# Instalar librerias para script de pendientes

sudo python -m pip install pandas

sudo python -m pip install plotnine

sudo python -m pip install sklearn

sudo python -m pip install openpyxl

# Instalar librerias para script de restaurante

sudo python -m pip install xlrd

sudo python -m pip install seaborn

sudo python -m pip install pandas_profiling

sudo python -m pip install pandas-profiling

sudo python -m !pip install -U pandas-profiling

sudo python -m pip install bs4

sudo yum install python3-devel

sudo python -m pip install tensorflow

sudo python -m pip install nltk

sudo python -m pip install spacy

sudo python -m pip install keras

sudo python -m pip install --upgrade pip

sudo python -m pip install --upgrade tensorflow-gpu
