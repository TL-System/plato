use park::ParkAlg;

fn main() {
    let addr = "127.0.0.1:4539".to_owned();
    portus::start!("netlink", None, ParkAlg(addr)).unwrap();
}
