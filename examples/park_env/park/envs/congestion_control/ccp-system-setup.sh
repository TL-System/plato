sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get -y install build-essential autoconf libtool libelf-dev 
# perf
sudo apt-get -y install linux-tools-common linux-tools-4.15.0-38-generic linux-tools-generic
# Mahimahi dependencies
sudo apt-get -y install autotools-dev dh-autoreconf iptables protobuf-compiler libprotobuf-dev pkg-config libssl-dev dnsmasq-base ssl-cert libxcb-present-dev libcairo2-dev libpango1.0-dev iproute2 apache2-dev apache2-bin
# iperf
sudo apt-get install -y iperf
# rust
curl https://sh.rustup.rs -sSf > rust.install.sh
chmod u+x ./rust.install.sh
chown vagrant:vagrant ./rust.install.sh
su -c "./rust.install.sh -y -v --default-toolchain nightly" vagrant
# Python setuptools
sudo apt-get -y install python3 python3-pip
sudo pip3 install --upgrade pip
# mahimahi setup
echo "sudo sysctl -w net.ipv4.ip_forward=1" >> ~/.bashrc

sudo pip install numpy networkx pycapnp
