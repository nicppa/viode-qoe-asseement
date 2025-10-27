#!/usr/bin/env python3
"""
Mininet minimal DASH demo (improved)
- starts a simple HTTP server on h1 serving /home/mininet/www
- auto-creates placeholder segments if missing
- prints interface mapping so you know which interface to capture on
- starts tcpdump on the correct client interface with -s 0
- performs curl requests to simulate playback
- stops tcpdump and reports capture path and size

Run with: sudo python3 mininet_minimal_dash.py
"""

from mininet.net import Mininet
from mininet.node import OVSController
from mininet.link import TCLink
from mininet.log import setLogLevel, info
import time
import os
import shlex

# Config
SERVER_DIR = '/var/html/out'
MPD_NAME = 'stream.mpd'
SEGMENT_PREFIX = 'chunk-stream0-0000'
SEGMENT_COUNT = 6
HTTP_PORT = 8000
PCAP_PATH = '/home/mininet/cn/h2_capture.pcap'
TCPDUMP_BIN = 'sudo tcpdump'  # assume in PATH


def ensure_server_files():
    """Ensure SERVER_DIR exists and has mpd + some segment files."""
    if not os.path.isdir(SERVER_DIR):
        os.makedirs(SERVER_DIR, exist_ok=True)
        info('Created server dir: %s\n' % SERVER_DIR)

    mpd_path = os.path.join(SERVER_DIR, MPD_NAME)
    # create placeholder mpd if missing
    if not os.path.isfile(mpd_path):
        with open(mpd_path, 'w') as f:
            f.write('<mpd>placeholder</mpd>')
        info('Wrote placeholder MPD: %s\n' % mpd_path)

    # create small random segments if missing
    for i in range(1, SEGMENT_COUNT + 1):
        seg = os.path.join(SERVER_DIR, f"{SEGMENT_PREFIX}{i}.m4s")
        if not os.path.isfile(seg):
            # small deterministic content to avoid /dev/urandom slowness
            with open(seg, 'wb') as f:
                f.write((f'SEGMENT-{i}\n').encode() * 1024)  # ~10KB-ish
            info('Wrote placeholder segment: %s\n' % seg)


def run():
    setLogLevel('info')
    ensure_server_files()

    net = Mininet(controller=OVSController, link=TCLink)
    c0 = net.addController('c0')
    h1 = net.addHost('h1', ip='10.0.0.1/24')   # server
    h2 = net.addHost('h2', ip='10.0.0.2/24')   # client
    s1 = net.addSwitch('s1')

    # Create links with default bw/delay. You can tune these.
    net.addLink(h1, s1, bw=10, delay='10ms', loss=0)
    net.addLink(h2, s1, bw=10, delay='10ms', loss=0)

    net.start()

    # 假设 h1 和 h2 已经在 net.addHost 中创建
    for host in [h1, h2]:
        print(f"Host {host.name} interfaces:")
        for intf in host.intfList():
            # intf.name 是接口名, host.IP(intf) 获取接口 IP
            print(f"  {intf.name} - {host.IP(intf=intf)}")

    info('\n*** Topology and interface mapping (use these names for tcpdump/tc)\n')
    info('Connections s1 <-> h2: %s\n' % str(s1.connectionsTo(h2)))
    # print interfaces on h2
    h2_ifaces = [intf.name for intf in h2.intfList()]
    info('h2 interfaces: %s\n' % h2_ifaces)

    # Determine client interface name (pick first non-loopback)
    client_iface = None
    for n in h2_ifaces:
        if n != 'lo':
            client_iface = n
            break
    if not client_iface:
        info('ERROR: could not determine h2 interface name.\n')
        net.stop()
        return

    info('Using client interface for capture: %s\n' % client_iface)

    # Start simple HTTP server on h1
    info('*** Starting HTTP server on h1 (serving %s)\n' % SERVER_DIR)
    # kill any prior python http.server on that port to avoid conflicts
    h1.cmd(f"pkill -f 'http.server {HTTP_PORT}' || true")
    # start server (redirect logs)
    h1.cmd(f'cd {shlex.quote(SERVER_DIR)} && python3 -m http.server {HTTP_PORT} > /tmp/http_server.log 2>&1 &')
    time.sleep(1)

    # sanity check: list server files
    server_ls = h1.cmd(f'ls -l {shlex.quote(SERVER_DIR)} | head -n 20')
    info('h1 server directory listing:\n%s\n' % server_ls)

    # Start tcpdump on h2 with full packet capture
    info('*** Starting tcpdump on h2 (full packet capture)\n')
    # Build tcpdump command: capture TCP port HTTP_PORT or all traffic
    tcpdump_cmd = f"{TCPDUMP_BIN} -i {client_iface} -s 0 -w {PCAP_PATH} tcp port {HTTP_PORT}"
    # run in background
    h2.cmd(tcpdump_cmd + ' &')
    time.sleep(0.5)

    # Confirm tcpdump started
    tcpdump_ps = h2.cmd("ps aux | grep '[t]cpdump' || true")
    info('tcpdump ps (h2):\n%s\n' % tcpdump_ps)

    # Now perform requests (mpd + segments)
    info('*** Simulating playback on h2 (curl requests)\n')
    h2.cmd(f'curl -s -o /dev/null http://10.0.0.1:{HTTP_PORT}/{MPD_NAME}')
    # small delay to ensure server-side logging
    time.sleep(0.2)

    for i in range(1, SEGMENT_COUNT + 1):
        fname = f"{SEGMENT_PREFIX}{i}.m4s"
        info('h2 requesting %s\n' % fname)
        res = h2.cmd(f'curl -s -o /dev/null -w "%{{http_code}}" http://10.0.0.1:{HTTP_PORT}/{fname}')
        info('h2 http code: %s\n' % res.strip())
        time.sleep(0.8)

    info('*** Requests finished; stopping tcpdump\n')
    # stop tcpdump gracefully by killing the tcpdump process in the h2 namespace
    h2.cmd('pkill -f tcpdump || true')
    time.sleep(0.5)

    # Check pcap presence and size
    pcap_info = h2.cmd(f'ls -lh {PCAP_PATH} || true')
    info('PCAP info (h2): %s\n' % pcap_info)

    # If pcap size is zero, show debug: tcpdump stdout tail and http server log
    try:
        size_str = pcap_info.split()[4]
    except Exception:
        size_str = '0'

    info('h1 http server log:\n%s\n' % h1.cmd('tail -n 20 /tmp/http_server.log'))

    if '0' in size_str or '0B' in size_str:
        info('Warning: pcap size is 0 or empty. Showing debug info:\n')
        http_log = h1.cmd('tail -n 50 /tmp/http_server.log || true')
        info('http server log:\n%s\n' % http_log)
        tcpdump_out = h2.cmd('dmesg | tail -n 50 || true')
        info('dmesg tail (h2):\n%s\n' % tcpdump_out)
    else:
        info('Capture saved at %s (inside Mininet h2 namespace)\n' % PCAP_PATH)

    net.stop()


if __name__ == '__main__':
    run()
