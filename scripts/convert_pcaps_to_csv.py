#!/usr/bin/env python3

import subprocess
from pathlib import Path


BASE_DIR = Path.home() / "sdn_dataset"
PCAP_DIR = BASE_DIR / "pcaps"
RAW_DIR = BASE_DIR / "raw_csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def convert_pcap(pcap_file):

    output_file = RAW_DIR / f"{pcap_file.stem}_raw.csv"

    print(f"[INFO] Converting: {pcap_file}")
    print(f"[INFO] Output    : {output_file}")

    cmd = [
        "tshark",
        "-r",
        str(pcap_file),
        "-Y",
        "ip && (tcp || udp)",
        "-T",
        "fields",
        "-e",
        "frame.time_epoch",
        "-e",
        "frame.len",
        "-e",
        "ip.src",
        "-e",
        "ip.dst",
        "-e",
        "ip.proto",
        "-e",
        "ip.ttl",
        "-e",
        "tcp.srcport",
        "-e",
        "tcp.dstport",
        "-e",
        "udp.srcport",
        "-e",
        "udp.dstport",
        "-e",
        "tcp.window_size",
        "-e",
        "tcp.len",
        "-E",
        "header=y",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        "-E",
        "occurrence=f"
    ]

    with open(output_file, "w") as f:
        subprocess.run(
            cmd,
            stdout=f,
            check=True
        )


def main():

    pcaps = sorted(PCAP_DIR.rglob("*.pcap"))

    print(f"[INFO] Found {len(pcaps)} PCAP files.")

    for pcap_file in pcaps:
        convert_pcap(pcap_file)

    print("[INFO] PCAP to CSV conversion completed.")


if __name__ == "__main__":
    main()
