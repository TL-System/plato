extern crate fnv;
extern crate portus;
extern crate tokio;

mod ccp_capnp;

use futures::Future;
use portus::ipc::Ipc;
use portus::lang::Scope;
use portus::DatapathTrait;
use std::collections::HashMap;

pub struct ParkAlg(pub String);

impl<I: Ipc> portus::CongAlg<I> for ParkAlg {
    type Flow = ParkFlow<I>;

    fn name() -> &'static str {
        "park"
    }

    fn datapath_programs(&self) -> HashMap<&'static str, String> {
        let mut h = HashMap::default();
        h.insert("park", include_str!("park.ccp").to_string());
        h
    }

    fn new_flow(&self, mut control: portus::Datapath<I>, i: portus::DatapathInfo) -> Self::Flow {
        let sc = control.set_program("park", None).unwrap();
        ParkFlow::init(sc, control, i, &self.0).unwrap()
    }
}

pub struct ParkFlow<I: Ipc> {
    sc: Scope,
    dp: portus::Datapath<I>,
    info: portus::DatapathInfo,
    runtime: tokio::runtime::current_thread::Runtime,
    client: ccp_capnp::r_l_agent::Client,
}

impl<I: Ipc> ParkFlow<I> {
    fn init(
        sc: Scope,
        dp: portus::Datapath<I>,
        info: portus::DatapathInfo,
        rl_server_addr: &str,
    ) -> std::io::Result<Self> {
        let mut runtime = tokio::runtime::current_thread::Runtime::new().unwrap();

        use std::net::ToSocketAddrs;
        let stream = runtime
            .block_on(tokio::net::TcpStream::connect(
                &rl_server_addr.to_socket_addrs().unwrap().next().unwrap(),
            ))
            .unwrap();

        use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
        use tokio::io::AsyncRead;

        let (reader, writer) = stream.split();

        let network = Box::new(twoparty::VatNetwork::new(
            reader,
            writer,
            rpc_twoparty_capnp::Side::Client,
            Default::default(),
        ));

        let mut rpc_system = RpcSystem::new(network, None);
        let client = rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);
        runtime.spawn(rpc_system.map_err(|_e| ()));

        Ok(ParkFlow {
            sc,
            dp,
            info,
            runtime,
            client,
        })
    }
}

impl<I: Ipc> portus::Flow for ParkFlow<I> {
    fn on_report(&mut self, _: u32, m: portus::Report) {
        // ask the RL agent what it wants to do
        let mut req = self.client.get_action_request();

        let mut obs = req.get().init_observation();
        obs.set_bytes_acked(m.get_field("Report.bytes_acked", &self.sc).unwrap());
        obs.set_bytes_misordered(m.get_field("Report.bytes_misordered", &self.sc).unwrap());
        obs.set_ecn_bytes(m.get_field("Report.ecn_bytes", &self.sc).unwrap());
        obs.set_packets_acked(m.get_field("Report.packets_acked", &self.sc).unwrap());
        obs.set_packets_misordered(m.get_field("Report.packets_misordered", &self.sc).unwrap());
        obs.set_ecn_packets(m.get_field("Report.ecn_packets", &self.sc).unwrap());
        obs.set_loss(m.get_field("Report.loss", &self.sc).unwrap());
        obs.set_timeout(m.get_field("Report.timeout", &self.sc).unwrap() != 0);
        obs.set_bytes_in_flight(m.get_field("Report.bytes_in_flight", &self.sc).unwrap());
        obs.set_packets_in_flight(m.get_field("Report.packets_in_flight", &self.sc).unwrap());
        obs.set_bytes_pending(m.get_field("Report.bytes_pending", &self.sc).unwrap());
        obs.set_rtt(m.get_field("Report.rtt", &self.sc).unwrap());
        obs.set_rin(m.get_field("Report.rin", &self.sc).unwrap());
        obs.set_rout(m.get_field("Report.rout", &self.sc).unwrap());

        let response = self.runtime.block_on(req.send().promise).unwrap();
        let action = response.get().unwrap().get_action().unwrap();
        let rate = match action.get_rate() {
            r if (r > 0) => Some(r),
            0 => None,
            _ => unreachable!(),
        };

        let cwnd = match action.get_cwnd() {
            c if (c > 0) => Some(c * self.info.mss),
            0 => None,
            _ => unreachable!(),
        };

        let updates: Vec<(&str, u32)> = rate
            .into_iter()
            .map(|r| ("Rate", r))
            .chain(cwnd.into_iter().map(|c| ("Cwnd", c)))
            .collect();

        if !updates.is_empty() {
            self.dp.update_field(&self.sc, &updates).unwrap();
        }
    }
}
