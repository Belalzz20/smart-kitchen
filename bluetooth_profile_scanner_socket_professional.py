import asyncio
import os
import socket
import traceback
from datetime import datetime

SOCKET_HOST = "127.0.0.1"
SOCKET_PORT = 5055
SOCKET_TIMEOUT_SECONDS = 3

# Scan timing
PYBLUEZ_SCAN_SECONDS = 5
BLEAK_SCAN_SECONDS = 4
SCAN_INTERVAL_SECONDS = 1

# Only logout after this many consecutive misses
MISS_BEFORE_LOGOUT = 3

LAST_SENT_KEY = None
MISSED_SCAN_COUNT = 0
CURRENT_PROFILE = ""
CURRENT_DEVICE_NAME = ""
CURRENT_ADDRESS = ""


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def candidate_paths(filename: str):
    here = script_dir()
    parent = os.path.dirname(here)
    cwd = os.getcwd()

    candidates = [
        os.path.join(here, filename),
        os.path.join(parent, filename),
        os.path.join(cwd, filename),
        os.path.join(os.path.dirname(cwd), filename),
        os.path.abspath(filename),
    ]

    seen = set()
    unique = []
    for p in candidates:
        p = os.path.abspath(p)
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def resolve_existing_file(filename: str):
    for path in candidate_paths(filename):
        if os.path.exists(path):
            return path
    return candidate_paths(filename)[0]


PROFILE_FILE = resolve_existing_file("SmartKitchenProfiles.txt")
OUTPUT_FILE = resolve_existing_file("active_bluetooth_device.txt")
VISIBLE_OUTPUT_FILE = resolve_existing_file("visible_bluetooth_devices.txt")


def clean_value(value):
    return (value or "").strip()


def normalize_text(value):
    return clean_value(value).lower()


def normalize_address(value):
    return normalize_text(value).replace("-", ":")


def parse_profiles(path):
    profiles = []

    if not os.path.exists(path):
        print(f"[PROFILE FILE NOT FOUND] {path}")
        return profiles

    current = None
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                if current:
                    profiles.append(current)

                current = {
                    "section": line[1:-1].strip(),
                    "name": "",
                    "bluetoothname": "",
                    "bluetoothaddress": "",
                    "profileorder": len(profiles),
                }
                continue

            if current is None or "=" not in line:
                continue

            key, value = line.split("=", 1)
            current[key.strip().lower()] = value.strip()

    if current:
        profiles.append(current)

    return profiles


def build_payload(detected, profile_name="", device_name="", address="", timestamp=""):
    return (
        f"Detected={str(bool(detected))};"
        f"Profile={clean_value(profile_name)};"
        f"DeviceName={clean_value(device_name)};"
        f"Address={clean_value(address)};"
        f"Timestamp={clean_value(timestamp)}"
    )


def write_presence_file(detected, profile_name="", device_name="", address=""):
    timestamp = datetime.utcnow().isoformat()
    payload = build_payload(detected, profile_name, device_name, address, timestamp)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(payload.replace(";", os.linesep))

    return payload


def write_visible_devices(lines):
    try:
        with open(VISIBLE_OUTPUT_FILE, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + os.linesep)
    except Exception as ex:
        print(f"[VISIBLE FILE WRITE ERROR] {ex}")


def send_to_csharp(payload):
    try:
        with socket.create_connection((SOCKET_HOST, SOCKET_PORT), timeout=SOCKET_TIMEOUT_SECONDS) as sock:
            sock.sendall((payload + "\n").encode("utf-8"))
            response = sock.recv(1024).decode("utf-8", errors="ignore").strip()
            print(f"C# reply: {response if response else 'NO_RESPONSE'}")
            return True
    except Exception as ex:
        print(f"Socket send failed: {ex}")
        return False


def publish_presence(detected, profile_name="", device_name="", address="", force=False):
    global LAST_SENT_KEY, CURRENT_PROFILE, CURRENT_DEVICE_NAME, CURRENT_ADDRESS

    timestamp = datetime.utcnow().isoformat()
    payload = build_payload(detected, profile_name, device_name, address, timestamp)
    state_key = build_payload(detected, profile_name, device_name, address, "")

    # refresh file every time so C# knows presence is still alive
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(payload.replace(";", os.linesep))

    if force or LAST_SENT_KEY != state_key:
        delivered = send_to_csharp(payload)
        LAST_SENT_KEY = state_key
        print(f"Published state: {state_key} | SocketDelivered: {delivered}")
    else:
        print(f"State unchanged (timestamp refreshed): {state_key}")

    if detected:
        CURRENT_PROFILE = clean_value(profile_name)
        CURRENT_DEVICE_NAME = clean_value(device_name)
        CURRENT_ADDRESS = clean_value(address)
    else:
        CURRENT_PROFILE = ""
        CURRENT_DEVICE_NAME = ""
        CURRENT_ADDRESS = ""


def scan_with_pybluez():
    visible = []
    visible_lines = []

    try:
        import bluetooth
        devices = bluetooth.discover_devices(lookup_names=True, duration=PYBLUEZ_SCAN_SECONDS)
    except Exception as ex:
        print(f"PyBluez scan failed: {ex}")
        return visible, visible_lines

    for address, name in devices:
        address = clean_value(address)
        name = clean_value(name)

        visible.append({
            "device_name": name,
            "address": address,
            "source": "PyBluez",
            "rssi": -100
        })

        # show all available device names only
        if name:
            print(f"Seen -> Name: {name} | Address: {address}")
            visible_lines.append(f"Name={name};Address={address};Source=PyBluez")

    return visible, visible_lines


async def scan_with_bleak():
    visible = []
    visible_lines = []

    try:
        from bleak import BleakScanner
        devices = await BleakScanner.discover(timeout=BLEAK_SCAN_SECONDS)
    except Exception as ex:
        print(f"Bleak scan failed: {ex}")
        return visible, visible_lines

    for device in devices:
        name = clean_value(getattr(device, "name", "") or "")
        address = clean_value(getattr(device, "address", "") or "")
        rssi = getattr(device, "rssi", -999)

        visible.append({
            "device_name": name,
            "address": address,
            "source": "Bleak",
            "rssi": rssi
        })

        # show all available device names only
        if name:
            print(f"Seen -> Name: {name} | Address: {address}")
            visible_lines.append(f"Name={name};Address={address};RSSI={rssi};Source=Bleak")

    return visible, visible_lines


def deduplicate_devices(devices):
    result = []
    seen = set()

    for dev in devices:
        key = (
            normalize_text(dev.get("device_name", "")),
            normalize_address(dev.get("address", ""))
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(dev)

    return result


def choose_best_match(discovered_devices, profiles):
    matches = []

    for dev_index, dev in enumerate(discovered_devices):
        device_name = normalize_text(dev.get("device_name", ""))
        address = normalize_address(dev.get("address", ""))

        for profile in profiles:
            p_name = normalize_text(profile.get("bluetoothname", ""))
            p_addr = normalize_address(profile.get("bluetoothaddress", ""))

            score = -1
            match_by = ""

            # 1) exact name first
            if device_name and p_name and device_name == p_name:
                score = 3
                match_by = "name-exact"

            # 2) contains name second
            elif device_name and p_name and p_name in device_name:
                score = 2
                match_by = "name-contains"

            # 3) address fallback
            elif address and p_addr and address == p_addr:
                score = 1
                match_by = "address"

            if score > 0:
                matches.append({
                    "profile_name": profile.get("name", ""),
                    "device_name": clean_value(dev.get("device_name", "")),
                    "address": clean_value(dev.get("address", "")),
                    "source": dev.get("source", ""),
                    "match_by": match_by,
                    "score": score,
                    "profile_order": profile.get("profileorder", 9999),
                    "device_order": dev_index,
                    "rssi": int(dev.get("rssi", -999) or -999)
                })

    matches.sort(
        key=lambda x: (
            -x["score"],         # name exact > name contains > address
            x["profile_order"],  # order in SmartKitchenProfiles.txt
            -x["rssi"],          # stronger signal first
            x["device_order"]    # earlier discovered
        )
    )

    return matches[0] if matches else None


async def scan_once():
    global MISSED_SCAN_COUNT

    profiles = parse_profiles(PROFILE_FILE)
    print(f"Profiles loaded: {len(profiles)} from {PROFILE_FILE}")

    if not profiles:
        MISSED_SCAN_COUNT = 0
        publish_presence(False, force=True)
        return

    visible_devices = []
    visible_lines = []

    # PyBluez first
    pybluez_devices, pybluez_lines = scan_with_pybluez()
    if pybluez_devices:
        print("Scanner: PyBluez")
    visible_devices.extend(pybluez_devices)
    visible_lines.extend(pybluez_lines)

    # Bleak second
    bleak_devices, bleak_lines = await scan_with_bleak()
    if bleak_devices:
        print("Scanner: Bleak")
    visible_devices.extend(bleak_devices)
    visible_lines.extend(bleak_lines)

    visible_devices = deduplicate_devices(visible_devices)
    write_visible_devices(visible_lines)

    best = choose_best_match(visible_devices, profiles)

    if best:
        MISSED_SCAN_COUNT = 0

        profile_name = best["profile_name"]
        device_name = best["device_name"]
        address = best["address"]

        print(
            f"Matched profile: {profile_name} | "
            f"Device: {device_name} | Address: {address} | "
            f"Source: {best['source']} | MatchBy: {best['match_by']}"
        )

        publish_presence(
            True,
            profile_name=profile_name,
            device_name=device_name,
            address=address
        )
    else:
        MISSED_SCAN_COUNT += 1
        print(f"No known Bluetooth profile detected. Miss count = {MISSED_SCAN_COUNT}/{MISS_BEFORE_LOGOUT}")

        # only logout after enough consecutive misses
        if MISSED_SCAN_COUNT >= MISS_BEFORE_LOGOUT:
            publish_presence(False)


async def main():
    print("Smart Kitchen Bluetooth profile scanner started...")
    print(f"Resolved profile file: {PROFILE_FILE}")
    print(f"Resolved active Bluetooth file: {OUTPUT_FILE}")
    print(f"Resolved visible devices file: {VISIBLE_OUTPUT_FILE}")
    print(f"Sending matches to C# socket server on {SOCKET_HOST}:{SOCKET_PORT}")
    print("Matching priority: exact name, then contains name, then address, then profile order.")
    print(f"Logout after {MISS_BEFORE_LOGOUT} missed scans.")

    # start in logged-out state
    publish_presence(False, force=True)

    while True:
        try:
            await scan_once()
        except Exception as ex:
            print(f"[SCAN LOOP ERROR] {ex}")
            traceback.print_exc()

        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())