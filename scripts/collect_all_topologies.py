
import argparse
import os
import subprocess
import time
from pathlib import Path

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.clean import cleanup


BASE_DIR = Path.home() / "sdn_dataset"

LINK_BW = 10
LINK_DELAY = "5ms"
LINK_LOSS = 0

IPERF_PORT = 5001

SCENARIOS = {
    "normal": {
        "rates": ["2M"],
        "label": 0
    },
    "moderate": {
        "rates": ["2M", "2M", "2M"],
        "label": 1
    },
    "heavy": {
        "rates": ["5M", "5M", "5M"],
        "label": 2
    }
}


def add_switch(net, name):
    return net.addSwitch(
        name,
        cls=OVSSwitch,
        failMode="standalone",
        stp=True
    )


def add_host(net, number):
    return net.addHost(
        f"h{number}",
        ip=f"10.0.0.{number}/24"
    )


def build_linear():
    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )

    h1 = add_host(net, 1)
    h2 = add_host(net, 2)
    h3 = add_host(net, 3)
    h4 = add_host(net, 4)

    s1 = add_switch(net, "s1")
    s2 = add_switch(net, "s2")
    s3 = add_switch(net, "s3")
    s4 = add_switch(net, "s4")

    cap_link = net.addLink(
        h1,
        s1,
        bw=LINK_BW,
        delay=LINK_DELAY,
        loss=LINK_LOSS
    )

    capture_intf = cap_link.intf2.name

    net.addLink(h2, s2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(h3, s3, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(h4, s4, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)

    net.addLink(s1, s2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(s2, s3, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(s3, s4, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)

    return net, h1, [h2, h3, h4], capture_intf


def build_ring():
    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )

    h1 = add_host(net, 1)
    h2 = add_host(net, 2)
    h3 = add_host(net, 3)
    h4 = add_host(net, 4)

    s1 = add_switch(net, "s1")
    s2 = add_switch(net, "s2")
    s3 = add_switch(net, "s3")
    s4 = add_switch(net, "s4")

    cap_link = net.addLink(
        h1,
        s1,
        bw=LINK_BW,
        delay=LINK_DELAY,
        loss=LINK_LOSS
    )

    capture_intf = cap_link.intf2.name

    net.addLink(h2, s2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(h3, s3, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(h4, s4, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)

    net.addLink(s1, s2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(s2, s3, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(s3, s4, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(s4, s1, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)

    return net, h1, [h2, h3, h4], capture_intf


def build_mesh():
    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )

    h1 = add_host(net, 1)
    h2 = add_host(net, 2)
    h3 = add_host(net, 3)
    h4 = add_host(net, 4)

    s1 = add_switch(net, "s1")
    s2 = add_switch(net, "s2")
    s3 = add_switch(net, "s3")
    s4 = add_switch(net, "s4")

    cap_link = net.addLink(
        h1,
        s1,
        bw=LINK_BW,
        delay=LINK_DELAY,
        loss=LINK_LOSS
    )

    capture_intf = cap_link.intf2.name

    net.addLink(h2, s2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(h3, s3, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(h4, s4, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)

    switches = [s1, s2, s3, s4]

    for i in range(len(switches)):
        for j in range(i + 1, len(switches)):
            net.addLink(
                switches[i],
                switches[j],
                bw=LINK_BW,
                delay=LINK_DELAY,
                loss=LINK_LOSS
            )

    return net, h1, [h2, h3, h4], capture_intf


def build_fattree():
    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )

    h1 = add_host(net, 1)
    h2 = add_host(net, 2)
    h3 = add_host(net, 3)
    h4 = add_host(net, 4)

    e1 = add_switch(net, "e1")
    e2 = add_switch(net, "e2")

    a1 = add_switch(net, "a1")
    a2 = add_switch(net, "a2")

    c1 = add_switch(net, "c1")
    c2 = add_switch(net, "c2")

    cap_link = net.addLink(
        h1,
        e1,
        bw=LINK_BW,
        delay=LINK_DELAY,
        loss=LINK_LOSS
    )

    capture_intf = cap_link.intf2.name

    net.addLink(h2, e1, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(h3, e2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(h4, e2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)

    net.addLink(e1, a1, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(e1, a2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(e2, a1, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(e2, a2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)

    net.addLink(a1, c1, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(a1, c2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(a2, c1, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)
    net.addLink(a2, c2, bw=LINK_BW, delay=LINK_DELAY, loss=LINK_LOSS)

    return net, h1, [h2, h3, h4], capture_intf


TOPO_BUILDERS = {
    "linear": build_linear,
    "ring": build_ring,
    "mesh": build_mesh,
    "fattree": build_fattree
}


def create_dirs():
    for topology in TOPO_BUILDERS.keys():
        for scenario in SCENARIOS.keys():
            path = BASE_DIR / "pcaps" / topology / scenario
            path.mkdir(parents=True, exist_ok=True)


def run_one_experiment(topology, scenario, run_id, duration):

    info(f"\n[INFO] Starting {topology} | {scenario} | run {run_id}\n")

    cleanup()

    builder = TOPO_BUILDERS[topology]

    net, dst_host, src_hosts, capture_intf = builder()

    net.start()

    info("[INFO] Waiting for STP/OVS convergence...\n")
    time.sleep(20)

    dst_host.cmd("killall -9 iperf 2>/dev/null")

    dst_host.cmd(
        f"iperf -s -u -p {IPERF_PORT} "
        f"> /tmp/iperf_server_{topology}_{scenario}_{run_id}.log 2>&1 &"
    )

    time.sleep(2)

    pcap_dir = BASE_DIR / "pcaps" / topology / scenario
    pcap_dir.mkdir(parents=True, exist_ok=True)

    pcap_file = (
        pcap_dir /
        f"{scenario}_{topology}_run{run_id:02d}.pcap"
    )

    tshark_cmd = [
        "tshark",
        "-i",
        capture_intf,
        "-f",
        "ip and (tcp or udp)",
        "-a",
        f"duration:{duration}",
        "-w",
        str(pcap_file)
    ]

    info(f"[INFO] Capture interface: {capture_intf}\n")
    info(f"[INFO] Saving PCAP: {pcap_file}\n")

    tshark_proc = subprocess.Popen(
        tshark_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(3)

    rates = SCENARIOS[scenario]["rates"]

    selected_hosts = src_hosts[-len(rates):]

    client_procs = []

    for host, rate in zip(selected_hosts, rates):

        cmd = (
            f"iperf -c {dst_host.IP()} "
            f"-u "
            f"-p {IPERF_PORT} "
            f"-b {rate} "
            f"-t {duration}"
        )

        info(f"[INFO] {host.name} running: {cmd}\n")

        proc = host.popen(cmd)
        client_procs.append(proc)

    for proc in client_procs:
        proc.wait()

    tshark_proc.wait()

    dst_host.cmd("killall -9 iperf 2>/dev/null")

    net.stop()

    cleanup()

    info(f"[INFO] Completed {topology} | {scenario} | run {run_id}\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60
    )

    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["linear", "ring", "mesh", "fattree"],
        choices=["linear", "ring", "mesh", "fattree"]
    )

    args = parser.parse_args()

    create_dirs()

    for topology in args.topologies:
        for run_id in range(1, args.runs + 1):
            for scenario in ["normal", "moderate", "heavy"]:
                run_one_experiment(
                    topology=topology,
                    scenario=scenario,
                    run_id=run_id,
                    duration=args.duration
                )

    print("\n[INFO] All PCAP data generation completed.")


if __name__ == "__main__":

    setLogLevel("info")
    main()
