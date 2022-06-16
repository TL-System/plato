import os

start_dir = os.getcwd()

# mahimahi
os.system("sudo sysctl -w net.ipv4.ip_forward=1")
os.system("sudo add-apt-repository -y ppa:keithw/mahimahi")
os.system("sudo apt-get -y update")
os.system("sudo apt-get -y install mahimahi")

# apache server
os.system("sudo apt-get -y install apache2")

# selenium
os.chdir( start_dir + '/park/envs/abr/' )
os.system("wget 'https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz'")
os.system("sudo apt-get -y install python-setuptools python-pip xvfb xserver-xephyr tightvncserver unzip")
os.system("tar xvzf selenium-2.39.0.tar.gz")
selenium_dir = start_dir + "/park/envs/abr/selenium-2.39.0"
os.chdir( selenium_dir )
os.system("pip3 install --user selenium")
os.system("sudo python setup.py install" )
os.system("sudo sh -c \"echo 'DBUS_SESSION_BUS_ADDRESS=/dev/null' > /etc/init.d/selenium\"")

# py virtual display
os.chdir( start_dir + '/park/envs/abr/')
os.system("pip install --user pyvirtualdisplay")
os.system("wget 'https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb' ")
os.system("sudo dpkg -i google-chrome-stable_current_amd64.deb")
os.system("sudo apt-get -f -y install")

# copy the webpage files to /var/www/html
os.chdir( start_dir )
os.system("sudo cp park/envs/abr/video_server/myindex_*.html /var/www/html")
os.system("sudo cp park/envs/abr/video_server/dash.all.min.js /var/www/html")
os.system("sudo cp -r park/envs/abr/video_server/video* /var/www/html")
os.system("sudo cp park/envs/abr/video_server/Manifest.mpd /var/www/html")
