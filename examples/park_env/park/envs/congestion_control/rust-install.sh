apt update
apt install -y build-essential capnproto curl
curl https://sh.rustup.rs -sSf > rustup.sh
sh rustup.sh -y --default-toolchain=nightly
source ~/.cargo/env
