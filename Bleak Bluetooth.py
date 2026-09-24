# pip install bleak
import asyncio
from bleak import BleakScanner
from datetime import datetime


async def scan_bluetooth_devices():
    print("Scanning for bluetooth devices...")
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        devices = await BleakScanner.discover()

        print(f"\nScan completed at {scan_time}")
        print(f"Found {len(devices)} devices:\n")

        for device in devices:
            print(f"Device Name: {device.name or 'Unknown'}")
            print(f"MAC Address: {device.address}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


async def main():
    print("Starting Bluetooth scan...")
    await scan_bluetooth_devices()


if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())
