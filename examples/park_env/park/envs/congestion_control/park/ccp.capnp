@0x97d8c9b4831fdcbb;

struct Observation {
    bytesAcked         @0   :UInt64;
    bytesMisordered    @1   :UInt64;
    ecnBytes           @2   :UInt64;
    packetsAcked       @3   :UInt64;
    packetsMisordered  @4   :UInt64;
    ecnPackets         @5   :UInt64;
    loss               @6   :UInt64;
    timeout            @7   :Bool;
    bytesInFlight      @8   :UInt64;
    packetsInFlight    @9   :UInt64;
    bytesPending       @10  :UInt64;
    rtt                @11  :UInt64;
    rin                @12  :UInt64;
    rout               @13  :UInt64;
}

struct Action {
    rate @0 :UInt32;
    cwnd @1 :UInt32;
}

interface RLAgent {
    getAction @0 (observation :Observation) -> (action :Action);
}
