extern crate capnpc;

fn main() {
    capnpc::CompilerCommand::new()
        .file("ccp.capnp")
        .edition(capnpc::RustEdition::Rust2018)
        .run()
        .expect("capnp compiler command");
}
