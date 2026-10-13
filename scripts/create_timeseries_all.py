#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path.home() / "sdn_dataset"
RAW_DIR = BASE_DIR / "raw_csv"
TS_DIR = BASE_DIR / "timeseries"

TS_DIR.mkdir(parents=True, exist_ok=True)

LINK_CAPACITY_MBPS = 10.0

RAW_COLUMNS = [
    "timestamp",
    "frame_len",
    "src_ip",
    "dst_ip",
    "protocol",
    "ttl",
    "tcp_src",
    "tcp_dst",
    "udp_src",
    "udp_dst",
    "tcp_window",
    "tcp_len"
]


OUTPUT_COLUMNS = [
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
    "label"
]


def get_label(filename):

    name = filename.lower()

    if name.startswith("normal"):
        return 0

    if name.startswith("moderate"):
        return 1

    if name.startswith("heavy"):
        return 2

    raise ValueError(f"Cannot detect label from filename: {filename}")


def read_raw_csv(path):

    df = pd.read_csv(
        path,
        low_memory=False,
        on_bad_lines="skip"
    )

    if len(df.columns) < len(RAW_COLUMNS):
        raise ValueError(
            f"{path} has fewer than {len(RAW_COLUMNS)} columns."
        )

    df = df.iloc[:, :len(RAW_COLUMNS)].copy()
    df.columns = RAW_COLUMNS

    return df


def clean_ip(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .replace(
            {
                "": "unknown",
                "nan": "unknown",
                "None": "unknown"
            }
        )
    )


def process_file(raw_file):

    label = get_label(raw_file.name)

    print(f"[INFO] Processing: {raw_file.name} | label={label}")

    df = read_raw_csv(raw_file)

    numeric_cols = [
        "timestamp",
        "frame_len",
        "protocol",
        "ttl",
        "tcp_src",
        "tcp_dst",
        "udp_src",
        "udp_dst",
        "tcp_window",
        "tcp_len"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["src_ip"] = clean_ip(df["src_ip"])
    df["dst_ip"] = clean_ip(df["dst_ip"])

    df = df.dropna(subset=["timestamp"])

    df["frame_len"] = df["frame_len"].fillna(0)
    df["protocol"] = df["protocol"].fillna(-1).astype(int)
    df["ttl"] = df["ttl"].fillna(0)

    df["second"] = df["timestamp"].astype(int)

    rows = []

    for sec, g in df.groupby("second", sort=True):

        packet_count = len(g)

        if packet_count == 0:
            continue

        byte_count = g["frame_len"].sum()

        throughput = (byte_count * 8) / 1_000_000

        packet_rate = packet_count

        avg_pkt_size = g["frame_len"].mean()
        max_pkt_size = g["frame_len"].max()
        min_pkt_size = g["frame_len"].min()

        std_pkt_size = g["frame_len"].std(ddof=0)

        if pd.isna(std_pkt_size):
            std_pkt_size = 0

        flow_columns = [
            "src_ip",
            "dst_ip",
            "protocol",
            "tcp_src",
            "tcp_dst",
            "udp_src",
            "udp_dst"
        ]

        flow_df = (
            g[flow_columns]
            .fillna(-1)
            .astype(str)
        )

        flow_count = flow_df.drop_duplicates().shape[0]

        unique_src_ips = g.loc[
            g["src_ip"] != "unknown",
            "src_ip"
        ].nunique()

        unique_dst_ips = g.loc[
            g["dst_ip"] != "unknown",
            "dst_ip"
        ].nunique()

        tcp_packets = len(g[g["protocol"] == 6])
        udp_packets = len(g[g["protocol"] == 17])

        tcp_ratio = (tcp_packets / packet_count) * 100
        udp_ratio = (udp_packets / packet_count) * 100

        avg_ttl = g["ttl"].mean()
        max_ttl = g["ttl"].max()
        min_ttl = g["ttl"].min()

        utilization = throughput / LINK_CAPACITY_MBPS
        utilization = min(utilization, 1.0)

        delay = (
            1
            + 20 * utilization
            + 80 * (utilization ** 2)
        )

        if utilization > 0.70:
            packet_loss = (utilization - 0.70) / 0.30
        else:
            packet_loss = 0

        packet_loss = min(max(packet_loss, 0), 1)

        rows.append(
            [
                sec,
                packet_count,
                byte_count,
                throughput,
                packet_rate,
                avg_pkt_size,
                max_pkt_size,
                min_pkt_size,
                std_pkt_size,
                flow_count,
                unique_src_ips,
                unique_dst_ips,
                tcp_packets,
                udp_packets,
                tcp_ratio,
                udp_ratio,
                avg_ttl,
                max_ttl,
                min_ttl,
                utilization,
                delay,
                packet_loss,
                label
            ]
        )

    out_df = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS
    )

    output_file = TS_DIR / raw_file.name.replace(
        "_raw.csv",
        "_timeseries.csv"
    )

    out_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"[INFO] Saved {output_file.name} | rows={len(out_df)}"
    )


def main():

    raw_files = sorted(RAW_DIR.glob("*_raw.csv"))

    print(f"[INFO] Found {len(raw_files)} raw CSV files.")

    for raw_file in raw_files:
        process_file(raw_file)

    print("[INFO] All time-series files created.")


if __name__ == "__main__":
    main()
