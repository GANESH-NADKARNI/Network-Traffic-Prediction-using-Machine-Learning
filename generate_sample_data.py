"""
generate_sample_data.py
Creates a realistic synthetic Wireshark hex-dump .txt file for testing.
Generates ~1000 packets in the expected format.

Usage:
    python generate_sample_data.py            # writes sample_capture.txt
    python generate_sample_data.py --n 5000   # 5000 packets
"""

import random
import argparse
import struct


PROTOCOLS = [
    ("TCP",     0x0800, 6),
    ("UDP",     0x0800, 17),
    ("ARP",     0x0806, 0),
    ("ICMPv6",  0x86DD, 58),
    ("UDP",     0x0800, 17),
]

WEIGHTS = [0.35, 0.28, 0.12, 0.10, 0.15]

SRC_MACS = [
    "6c:0b:5e:46:ca:0e",
    "c8:d9:d2:29:62:7b",
    "d4:c1:9e:02:71:c0",
    "c8:5a:cf:05:d4:ec",
]

def mac_to_bytes(mac: str) -> list:
    return [int(x, 16) for x in mac.split(":")]

def ip_to_bytes(ip: str) -> list:
    return [int(x) for x in ip.split(".")]

def build_arp_packet(src_mac, src_ip, dst_ip) -> list:
    dst = [0xff]*6
    src = mac_to_bytes(src_mac)
    eth = dst + src + [0x08, 0x06]
    arp = [
        0x00,0x01, 0x08,0x00, 0x06, 0x04, 0x00,0x01
    ] + src + ip_to_bytes(src_ip) + [0x00]*6 + ip_to_bytes(dst_ip)
    return eth + arp + [0x00]*18

def build_ipv4_tcp_packet(src_mac, src_ip, dst_ip,
                           sport=12345, dport=80, payload_len=40) -> list:
    src_m = mac_to_bytes(src_mac)
    dst_m = [0x00,0x11,0x22,0x33,0x44,0x55]
    eth   = dst_m + src_m + [0x08, 0x00]
    total_len = 20 + 20 + payload_len
    ip = [
        0x45, 0x00,
        (total_len >> 8) & 0xFF, total_len & 0xFF,
        0x00, 0x01, 0x40, 0x00, 0x40, 0x06,
        0x00, 0x00,
    ] + ip_to_bytes(src_ip) + ip_to_bytes(dst_ip)
    tcp = [
        (sport >> 8)&0xFF, sport&0xFF,
        (dport >> 8)&0xFF, dport&0xFF,
        0x00,0x00,0x00,0x01,
        0x00,0x00,0x00,0x00,
        0x50,0x18, 0xFF,0xFF, 0x00,0x00, 0x00,0x00,
    ]
    payload = [random.randint(0,255) for _ in range(payload_len)]
    return eth + ip + tcp + payload

def build_ipv4_udp_packet(src_mac, src_ip, dst_ip,
                           sport=5353, dport=5353, payload_len=30) -> list:
    src_m = mac_to_bytes(src_mac)
    dst_m = [0x01,0x00,0x5e,0x00,0x00,0xfb]
    eth   = dst_m + src_m + [0x08, 0x00]
    total_len = 20 + 8 + payload_len
    ip = [
        0x45, 0x00,
        (total_len >> 8)&0xFF, total_len&0xFF,
        0x00,0x01,0x40,0x00, 0x01,0x11,
        0x00,0x00,
    ] + ip_to_bytes(src_ip) + ip_to_bytes(dst_ip)
    udp_len = 8 + payload_len
    udp = [
        (sport>>8)&0xFF, sport&0xFF,
        (dport>>8)&0xFF, dport&0xFF,
        (udp_len>>8)&0xFF, udp_len&0xFF,
        0x00,0x00,
    ]
    payload = [random.randint(0,255) for _ in range(payload_len)]
    return eth + ip + udp + payload

def bytes_to_hex_line(raw: list) -> str:
    parts = [f"{b:02x}" for b in raw]
    return "|0   |" + "|".join(parts) + "|"

def generate_txt(n_packets: int, outfile: str):
    random.seed(42)
    subnets = ["10.12.3", "10.12.2", "169.254.108"]

    ts_ms = 11*3600_000 + 5*60_000 + 55_000  # 11:05:55.000

    lines = []
    for i in range(n_packets):
        # Advance time by exponential inter-arrival (mean ~2ms)
        ts_ms += max(1, int(random.expovariate(1/2)))

        h  = (ts_ms // 3_600_000) % 24
        m  = (ts_ms //   60_000) % 60
        s  = (ts_ms //    1_000) % 60
        ms = ts_ms % 1000
        us = random.randint(0, 999)
        ts_str = f"{h:02d}:{m:02d}:{s:02d},{ms:03d},{us:03d}"

        src_mac = random.choice(SRC_MACS)
        src_ip  = f"{random.choice(subnets)}.{random.randint(1,254)}"
        dst_ip  = f"{random.choice(subnets)}.{random.randint(1,254)}"

        proto_name, eth_type, ip_proto = random.choices(PROTOCOLS, weights=WEIGHTS)[0]

        if eth_type == 0x0806:
            raw = build_arp_packet(src_mac, src_ip, dst_ip)
        elif ip_proto == 6:
            sport = random.randint(1024, 65535)
            dport = random.choice([80, 443, 8080, 22, 3389])
            plen  = random.randint(20, 1400)
            raw = build_ipv4_tcp_packet(src_mac, src_ip, dst_ip, sport, dport, plen)
        elif ip_proto == 17:
            sport = random.choice([5353, 137, 138, 1900, 5355])
            dport = sport
            plen  = random.randint(10, 500)
            raw = build_ipv4_udp_packet(src_mac, src_ip, dst_ip, sport, dport, plen)
        else:
            # Minimal ICMPv6-like
            raw = [0x33,0x33,0x00,0x00,0x00,0x16] + mac_to_bytes(src_mac) + [0x86,0xDD] + [0x60] + [0x00]*79

        lines.append("+---------+---------------+----------+")
        lines.append(f"{ts_str}   ETHER")
        lines.append(bytes_to_hex_line(raw))
        lines.append("")

    lines.append("+---------+---------------+----------+")

    with open(outfile, "w") as f:
        f.write("\n".join(lines))

    print(f"✔ Generated {n_packets} packets → {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of packets to generate")
    parser.add_argument("--out", default="sample_capture.txt",
                        help="Output file path")
    args = parser.parse_args()
    generate_txt(args.n, args.out)
