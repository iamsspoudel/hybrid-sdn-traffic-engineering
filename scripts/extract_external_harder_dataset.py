#!/usr/bin/env python3

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "timestamp",
    "packet_count",
    "byte_count",
    "throughput",
    "packet_rate",
    "avg_pkt_size",
    "max_pkt_size",
    "min_pkt_size",
    "std_pkt_size",
    "flow_count",
    "unique_src_ips",
    "unique_dst_ips",
    "tcp_packets",
    "udp_packets",
    "tcp_ratio",
    "udp_ratio",
    "avg_ttl",
    "max_ttl",
    "min_ttl",
    "utilization",
    "delay",
    "packet_loss",
    "label",
    "topology",
    "source_file"
]


def stable_seed(text):
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def detect_label(path_text):
    text = path_text.lower()

    if "normal" in text:
        return 0

    if "moderate" in text:
        return 1

    if "heavy" in text:
        return 2

    raise ValueError(f"Could not detect label from: {path_text}")


def detect_topology(path_text):
    text = path_text.lower()

    if "linear" in text:
        return "Linear"

    if "ring" in text:
        return "Ring"

    if "mesh" in text:
        return "Mesh"

    if "fattree" in text or "fat_tree" in text or "fat-tree" in text:
        return "Fat-tree"

    return "Unknown"


def pcap_to_raw_dataframe(pcap_file):
    fields = [
        "frame.time_epoch",
        "frame.len",
        "ip.src",
        "ip.dst",
        "ip.proto",
        "ip.ttl",
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport"
    ]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_csv = Path(tmp.name)

    cmd = [
        "tshark",
        "-r", str(pcap_file),
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
    ]

    for field in fields:
        cmd.extend(["-e", field])

    with open(tmp_csv, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=False)

    try:
        df = pd.read_csv(tmp_csv)
    except Exception:
        df = pd.read_csv(tmp_csv, engine="python", on_bad_lines="skip")

    tmp_csv.unlink(missing_ok=True)

    df.columns = [
        "time",
        "frame_len",
        "ip_src",
        "ip_dst",
        "ip_proto",
        "ip_ttl",
        "tcp_srcport",
        "tcp_dstport",
        "udp_srcport",
        "udp_dstport"
    ]

    return df


def estimate_delay_and_loss(utilization, label, rng):
    utilization = float(utilization)

    delay = 5.0 + 0.45 * utilization + max(0.0, utilization - 60.0) * 0.75
    delay += rng.normal(0, 8)

    if label == 0:
        delay += rng.normal(0, 6)
        loss_threshold = rng.normal(90, 8)
    elif label == 1:
        delay += rng.normal(8, 8)
        loss_threshold = rng.normal(78, 10)
    else:
        delay += rng.normal(18, 10)
        loss_threshold = rng.normal(66, 12)

    delay = max(1.0, delay)

    packet_loss = max(0.0, (utilization - loss_threshold) / 100.0)
    packet_loss += rng.normal(0, 0.015)
    packet_loss = float(np.clip(packet_loss, 0.0, 1.0))

    return delay, packet_loss


def aggregate_raw_packets(raw_df, label, topology, source_file, window_sec=1, capacity_mbps=10.0):
    df = raw_df.copy()

    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["frame_len"] = pd.to_numeric(df["frame_len"], errors="coerce")
    df["ip_proto"] = pd.to_numeric(df["ip_proto"], errors="coerce")
    df["ip_ttl"] = pd.to_numeric(df["ip_ttl"], errors="coerce")

    df = df.dropna(subset=["time", "frame_len"])

    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df["relative_time"] = df["time"] - df["time"].min()
    df["window"] = (df["relative_time"] // window_sec).astype(int)

    for col in ["tcp_srcport", "tcp_dstport", "udp_srcport", "udp_dstport"]:
        df[col] = df[col].fillna("").astype(str)

    df["src_port"] = np.where(df["tcp_srcport"] != "", df["tcp_srcport"], df["udp_srcport"])
    df["dst_port"] = np.where(df["tcp_dstport"] != "", df["tcp_dstport"], df["udp_dstport"])

    df["flow_id"] = (
        df["ip_src"].astype(str) + "_" +
        df["ip_dst"].astype(str) + "_" +
        df["ip_proto"].astype(str) + "_" +
        df["src_port"].astype(str) + "_" +
        df["dst_port"].astype(str)
    )

    rows = []

    rng = np.random.default_rng(stable_seed(source_file))

    for window_id, group in df.groupby("window"):
        packet_count = len(group)
        byte_count = group["frame_len"].sum()

        throughput = (byte_count * 8.0) / (window_sec * 1_000_000.0)
        packet_rate = packet_count / window_sec

        utilization = (throughput / capacity_mbps) * 100.0
        utilization = float(np.clip(utilization, 0.0, 130.0))

        tcp_packets = int((group["ip_proto"] == 6).sum())
        udp_packets = int((group["ip_proto"] == 17).sum())

        tcp_ratio = tcp_packets / packet_count if packet_count > 0 else 0.0
        udp_ratio = udp_packets / packet_count if packet_count > 0 else 0.0

        delay, packet_loss = estimate_delay_and_loss(utilization, label, rng)

        ttl_values = group["ip_ttl"].dropna()

        if ttl_values.empty:
            avg_ttl = 0.0
            max_ttl = 0.0
            min_ttl = 0.0
        else:
            avg_ttl = ttl_values.mean()
            max_ttl = ttl_values.max()
            min_ttl = ttl_values.min()

        row = {
            "timestamp": int(window_id),
            "packet_count": packet_count,
            "byte_count": byte_count,
            "throughput": throughput,
            "packet_rate": packet_rate,
            "avg_pkt_size": group["frame_len"].mean(),
            "max_pkt_size": group["frame_len"].max(),
            "min_pkt_size": group["frame_len"].min(),
            "std_pkt_size": group["frame_len"].std(ddof=0),
            "flow_count": group["flow_id"].nunique(),
            "unique_src_ips": group["ip_src"].nunique(),
            "unique_dst_ips": group["ip_dst"].nunique(),
            "tcp_packets": tcp_packets,
            "udp_packets": udp_packets,
            "tcp_ratio": tcp_ratio,
            "udp_ratio": udp_ratio,
            "avg_ttl": avg_ttl,
            "max_ttl": max_ttl,
            "min_ttl": min_ttl,
            "utilization": utilization,
            "delay": delay,
            "packet_loss": packet_loss,
            "label": label,
            "topology": topology,
            "source_file": source_file
        }

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pcap_dir", type=str, default="/root/sdn_external_harder/pcaps")
    parser.add_argument("--output_csv", type=str, default="/root/sdn_external_harder/external_harder_dataset.csv")
    parser.add_argument("--capacity_mbps", type=float, default=10.0)

    args = parser.parse_args()

    pcap_dir = Path(args.pcap_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pcap_files = sorted(pcap_dir.rglob("*.pcap"))

    if not pcap_files:
        raise FileNotFoundError(f"No PCAP files found in {pcap_dir}")

    print(f"[INFO] Found PCAP files: {len(pcap_files)}")

    all_dfs = []

    for pcap_file in pcap_files:
        print(f"[INFO] Processing: {pcap_file}")

        label = detect_label(str(pcap_file))
        topology = detect_topology(str(pcap_file))

        raw_df = pcap_to_raw_dataframe(pcap_file)

        agg_df = aggregate_raw_packets(
            raw_df=raw_df,
            label=label,
            topology=topology,
            source_file=pcap_file.name,
            capacity_mbps=args.capacity_mbps
        )

        if not agg_df.empty:
            all_dfs.append(agg_df)

    if not all_dfs:
        raise RuntimeError("No usable external dataset rows were generated.")

    final_df = pd.concat(all_dfs, ignore_index=True)

    final_df = final_df[FEATURE_COLUMNS]
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    final_df = final_df.fillna(0)

    final_df.to_csv(output_csv, index=False)

    print("\n[INFO] External harder dataset saved:")
    print(output_csv)

    print("\n[INFO] Shape:", final_df.shape)

    print("\n========== LABEL DISTRIBUTION ==========")
    print(final_df["label"].value_counts().sort_index())

    print("\n========== TOPOLOGY DISTRIBUTION ==========")
    print(final_df["topology"].value_counts())

    print("\n========== MEAN FEATURES BY LABEL ==========")
    print(final_df.groupby("label")[["throughput", "utilization", "delay", "packet_loss"]].mean())


if __name__ == "__main__":
    main()
