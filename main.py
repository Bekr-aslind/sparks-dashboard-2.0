import os
import csv
import math
import time
import random
import base64
import shutil
import subprocess
from datetime import datetime

import cv2
import qrcode
import requests


# =========================================================
# S.P.A.R.K.S MAIN SYSTEM CODE
# Sampling Prediction and Risk Knowledge System
# =========================================================


# =========================================================
# MODE SETTINGS
# =========================================================
# Set this to True when running on Raspberry Pi with real hardware.
# Set to False only if you want to test the code on a laptop/Windows without hardware.
RUN_WITH_REAL_HARDWARE = False

# If YOLO model is not ready yet, leave this empty.
# Later example:
# YOLO_MODEL_PATH = "best.pt"
YOLO_MODEL_PATH = "INSERT_YOUR_YOLO_MODEL_PATH_HERE"

# Optional Google Apps Script URL.
# Later example:
# GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/XXXXX/exec"
GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbxI4j2BA0OzR1xn7qXnF87s9L2mpq_1zdAm96qeYnaEHsToa1MazUdNZT1U1vs6JGe1-g/exec"

# Optional online dashboard link.
# Later example:
# DASHBOARD_BASE_URL = "https://your-dashboard-link"
DASHBOARD_BASE_URL = "INSERT_YOUR_ONLINE_DASHBOARD_LINK_HERE"


# =========================================================
# PROJECT FOLDERS
# =========================================================
DATA_DIR = "data"
CAPTURES_DIR = "captures"
PROCESSED_DIR = "processed"
QR_DIR = "qr_codes"

CSV_FILE = os.path.join(DATA_DIR, "inspection_results.csv")
COUNTER_FILE = os.path.join(DATA_DIR, "counter.txt")


# =========================================================
# CAMERA SETTINGS
# =========================================================
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960


# =========================================================
# GPIO PIN SETTINGS - BCM NUMBERING
# =========================================================
# IR sensors
IR_VISUAL_GPIO = 5       # Physical Pin 29
IR_DC_GPIO = 6           # Physical Pin 31

# If your IR sensor logic is reversed, change this to 0.
IR_DETECTED_VALUE = 1

# L298N motor driver
MOTOR_IN1_GPIO = 23      # Physical Pin 16
MOTOR_IN2_GPIO = 24      # Physical Pin 18
MOTOR_ENA_GPIO = 18      # Physical Pin 12, PWM

MOTOR_SPEED = 0.65       # 0.0 to 1.0

# Servos
SERVO_SORT_GOOD_GPIO = 12        # Physical Pin 32
SERVO_SORT_DEFECT_GPIO = 13      # Physical Pin 33
SERVO_PLATFORM_GPIO = 26         # Physical Pin 37

# Servo angles
SORT_SERVO_NEUTRAL_ANGLE = 0
SORT_SERVO_ACTIVE_ANGLE = 60

PLATFORM_STANDBY_ANGLE = 0
PLATFORM_LOWER_ANGLE = 70

SERVO_SETTLE_TIME = 0.6


# =========================================================
# SEQUENCE SETTINGS
# =========================================================
IR_WAIT_TIMEOUT = 30
DC_STABILIZE_TIME = 0.5
ADS_SAMPLE_COUNT = 10
ADS_SAMPLE_DELAY = 0.05

SORTING_TRAVEL_TIME = 1.5
SORT_SERVO_HOLD_TIME = 0.8


# =========================================================
# DC VOLTAGE TEST SETTINGS
# =========================================================
# LM358 voltage follower expected output should follow reference voltage.
EXPECTED_VOLTAGE_MIN = 2.30
EXPECTED_VOLTAGE_MAX = 2.70

# ADS1115 channel
# Latest update: LM358 Pin 1 output -> ADS1115 A2
ADS1115_CHANNEL = "A2"


# =========================================================
# YOLO CLASS SETTINGS
# =========================================================
# Train your YOLO model with classes similar to:
# base / base_pad / substrate / reference
# die / top_die / top_square
BASE_CLASS_KEYWORDS = ["base", "pad", "substrate", "reference"]
DIE_CLASS_KEYWORDS = ["die", "top"]


# =========================================================
# CSV HEADER
# =========================================================
CSV_HEADER = [
    "tray_id",
    "die_id",
    "image_name",
    "base_detected",
    "die_detected",
    "base_confidence",
    "die_confidence",
    "base_bbox",
    "die_bbox",
    "x_offset_px",
    "y_offset_px",
    "offset_distance_px",
    "offset_percent",
    "overlap_percent",
    "misalignment_percent",
    "visual_status",
    "visual_score",
    "voltage_value",
    "electrical_status",
    "final_die_score",
    "tray_risk",
    "timestamp",
    "image_url",
    "qr_code_url"
]


# =========================================================
# BASIC UTILITY FUNCTIONS
# =========================================================
def is_placeholder(value):
    if value is None:
        return True

    value = str(value).strip()

    if value == "":
        return True

    if "INSERT_YOUR" in value:
        return True

    return False


def ensure_project_folders():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(QR_DIR, exist_ok=True)


def create_csv_if_missing():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_HEADER)
            writer.writeheader()


def load_counter():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if content:
            tray_counter, die_counter = content.split(",")
            return int(tray_counter), int(die_counter)

    return 1, 1


def save_counter(tray_counter, die_counter):
    with open(COUNTER_FILE, "w", encoding="utf-8") as file:
        file.write(f"{tray_counter},{die_counter}")


def append_row_to_csv(row):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADER)
        writer.writerow(row)


def file_to_base64(file_path):
    if not file_path or not os.path.exists(file_path):
        return ""

    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def bbox_to_string(bbox):
    if bbox is None:
        return ""

    return ",".join(str(round(float(value), 2)) for value in bbox)


# =========================================================
# HARDWARE SETUP
# =========================================================
def setup_hardware():
    if not RUN_WITH_REAL_HARDWARE:
        print("Running without real hardware. Hardware actions will be simulated.")
        return {
            "ir_visual": None,
            "ir_dc": None,
            "motor_in1": None,
            "motor_in2": None,
            "motor_ena": None,
            "servo_good": None,
            "servo_defect": None,
            "servo_platform": None,
            "ads_channel": None
        }

    print("Setting up Raspberry Pi hardware...")

    from gpiozero import DigitalInputDevice, OutputDevice, PWMOutputDevice, AngularServo

    # IR sensors
    ir_visual = DigitalInputDevice(IR_VISUAL_GPIO, pull_up=False)
    ir_dc = DigitalInputDevice(IR_DC_GPIO, pull_up=False)

    # Motor driver
    motor_in1 = OutputDevice(MOTOR_IN1_GPIO)
    motor_in2 = OutputDevice(MOTOR_IN2_GPIO)
    motor_ena = PWMOutputDevice(MOTOR_ENA_GPIO)

    # Servos
    servo_good = AngularServo(
        SERVO_SORT_GOOD_GPIO,
        min_angle=-90,
        max_angle=90,
        min_pulse_width=0.0005,
        max_pulse_width=0.0025
    )

    servo_defect = AngularServo(
        SERVO_SORT_DEFECT_GPIO,
        min_angle=-90,
        max_angle=90,
        min_pulse_width=0.0005,
        max_pulse_width=0.0025
    )

    servo_platform = AngularServo(
        SERVO_PLATFORM_GPIO,
        min_angle=-90,
        max_angle=90,
        min_pulse_width=0.0005,
        max_pulse_width=0.0025
    )

    # ADS1115
    try:
        import board
        import busio
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn

        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)
        ads.gain = 1

        # LM358 Pin 1 output is connected to ADS1115 A2
        ads_channel = AnalogIn(ads, ADS.P2)

    except Exception as error:
        print("WARNING: ADS1115 setup failed.")
        print("Reason:", error)
        print("Voltage reading will be simulated until ADS1115 is fixed.")
        ads_channel = None

    hardware = {
        "ir_visual": ir_visual,
        "ir_dc": ir_dc,
        "motor_in1": motor_in1,
        "motor_in2": motor_in2,
        "motor_ena": motor_ena,
        "servo_good": servo_good,
        "servo_defect": servo_defect,
        "servo_platform": servo_platform,
        "ads_channel": ads_channel
    }

    motor_stop(hardware)
    reset_all_servos(hardware)

    print("Hardware setup completed.")
    return hardware


# =========================================================
# MOTOR FUNCTIONS
# =========================================================
def motor_forward(hardware):
    print("Conveyor motor: FORWARD")

    if not RUN_WITH_REAL_HARDWARE:
        return

    hardware["motor_in1"].on()
    hardware["motor_in2"].off()
    hardware["motor_ena"].value = MOTOR_SPEED


def motor_stop(hardware):
    print("Conveyor motor: STOP")

    if not RUN_WITH_REAL_HARDWARE:
        return

    hardware["motor_ena"].value = 0
    hardware["motor_in1"].off()
    hardware["motor_in2"].off()


def move_motor_for_seconds(hardware, seconds):
    motor_forward(hardware)
    time.sleep(seconds)
    motor_stop(hardware)


# =========================================================
# SERVO FUNCTIONS
# =========================================================
def set_servo_angle(servo, angle):
    if RUN_WITH_REAL_HARDWARE and servo is not None:
        servo.angle = angle

    time.sleep(SERVO_SETTLE_TIME)


def reset_all_servos(hardware):
    print("Resetting servos to standby position...")

    set_servo_angle(hardware["servo_good"], SORT_SERVO_NEUTRAL_ANGLE)
    set_servo_angle(hardware["servo_defect"], SORT_SERVO_NEUTRAL_ANGLE)
    set_servo_angle(hardware["servo_platform"], PLATFORM_STANDBY_ANGLE)


def lower_dc_platform(hardware):
    print("Lowering DC test platform...")
    set_servo_angle(hardware["servo_platform"], PLATFORM_LOWER_ANGLE)


def raise_dc_platform(hardware):
    print("Raising DC test platform...")
    set_servo_angle(hardware["servo_platform"], PLATFORM_STANDBY_ANGLE)


def sort_sample(hardware, final_status):
    """
    final_status:
    GOOD    -> activate good sorting servo
    WARNING -> neutral/middle path
    DEFECT  -> activate defect sorting servo
    """

    final_status = str(final_status).upper()

    if final_status == "GOOD":
        print("Sorting sample to GOOD bin...")
        set_servo_angle(hardware["servo_good"], SORT_SERVO_ACTIVE_ANGLE)
        time.sleep(SORT_SERVO_HOLD_TIME)
        set_servo_angle(hardware["servo_good"], SORT_SERVO_NEUTRAL_ANGLE)

    elif final_status == "WARNING":
        print("Sorting sample to WARNING bin...")
        print("Using neutral/middle path for WARNING bin.")
        time.sleep(SORT_SERVO_HOLD_TIME)

    else:
        print("Sorting sample to DEFECT bin...")
        set_servo_angle(hardware["servo_defect"], SORT_SERVO_ACTIVE_ANGLE)
        time.sleep(SORT_SERVO_HOLD_TIME)
        set_servo_angle(hardware["servo_defect"], SORT_SERVO_NEUTRAL_ANGLE)


# =========================================================
# IR SENSOR FUNCTIONS
# =========================================================
def wait_for_ir_detection(hardware, sensor_name, timeout=IR_WAIT_TIMEOUT):
    print(f"Waiting for {sensor_name} detection...")

    if not RUN_WITH_REAL_HARDWARE:
        time.sleep(1)
        print(f"{sensor_name} simulated detection.")
        return True

    sensor = hardware[sensor_name]
    start_time = time.time()

    while True:
        if sensor.value == IR_DETECTED_VALUE:
            print(f"{sensor_name} detected test subject.")
            return True

        if time.time() - start_time > timeout:
            print(f"Timeout waiting for {sensor_name}.")
            return False

        time.sleep(0.05)


# =========================================================
# CAMERA CAPTURE
# =========================================================
def capture_image(image_path):
    print("Capturing image...")

    if not RUN_WITH_REAL_HARDWARE:
        import numpy as np

        image = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 255, dtype=np.uint8)

        cv2.putText(
            image,
            "SIMULATED IMAGE",
            (60, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            3
        )

        cv2.imwrite(image_path, image)
        print("Simulated image created:", image_path)
        return image_path

    camera_commands = [
        [
            "rpicam-still",
            "-n",
            "-o",
            image_path,
            "--width",
            str(CAMERA_WIDTH),
            "--height",
            str(CAMERA_HEIGHT)
        ],
        [
            "libcamera-still",
            "-n",
            "-o",
            image_path,
            "--width",
            str(CAMERA_WIDTH),
            "--height",
            str(CAMERA_HEIGHT)
        ]
    ]

    for command in camera_commands:
        try:
            subprocess.run(command, check=True)
            print("Image captured:", image_path)
            return image_path

        except FileNotFoundError:
            continue

        except subprocess.CalledProcessError as error:
            print("Camera command failed:", error)
            continue

    raise RuntimeError("Camera capture failed. Check rpicam-still/libcamera-still installation.")

# =========================================================
# YOLO MODEL SETUP
# =========================================================
def load_yolo_model():
    if is_placeholder(YOLO_MODEL_PATH):
        print("YOLO model path is empty. Visual inspection will use simulated boxes for now.")
        return None

    if not os.path.exists(YOLO_MODEL_PATH):
        print("YOLO model file not found:", YOLO_MODEL_PATH)
        print("Visual inspection will use simulated boxes for now.")
        return None

    print("Loading YOLO model...")

    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded.")
    return model


# =========================================================
# ALIGNMENT CALCULATION
# =========================================================
def calculate_alignment(base_bbox, die_bbox):
    base_x1, base_y1, base_x2, base_y2 = base_bbox
    die_x1, die_y1, die_x2, die_y2 = die_bbox

    base_center_x = (base_x1 + base_x2) / 2
    base_center_y = (base_y1 + base_y2) / 2

    die_center_x = (die_x1 + die_x2) / 2
    die_center_y = (die_y1 + die_y2) / 2

    x_offset_px = die_center_x - base_center_x
    y_offset_px = die_center_y - base_center_y

    offset_distance_px = math.sqrt((x_offset_px ** 2) + (y_offset_px ** 2))

    base_width = base_x2 - base_x1
    base_height = base_y2 - base_y1
    reference_size = min(base_width, base_height)

    if reference_size <= 0:
        offset_percent = 100
    else:
        offset_percent = (offset_distance_px / reference_size) * 100

    # Area overlap calculation
    overlap_x1 = max(base_x1, die_x1)
    overlap_y1 = max(base_y1, die_y1)
    overlap_x2 = min(base_x2, die_x2)
    overlap_y2 = min(base_y2, die_y2)

    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)
    overlap_area = overlap_width * overlap_height

    die_width = die_x2 - die_x1
    die_height = die_y2 - die_y1
    die_area = die_width * die_height

    if die_area <= 0:
        overlap_percent = 0
    else:
        overlap_percent = (overlap_area / die_area) * 100

    area_misalignment_percent = 100 - overlap_percent

    # Overall misalignment index:
    # This uses the larger value between center shift percentage and outside-area percentage.
    # This helps because the die can still be inside the base area but shifted from the center.
    misalignment_percent = max(offset_percent, area_misalignment_percent)

    return {
        "x_offset_px": round(x_offset_px, 2),
        "y_offset_px": round(y_offset_px, 2),
        "offset_distance_px": round(offset_distance_px, 2),
        "offset_percent": round(offset_percent, 2),
        "overlap_percent": round(overlap_percent, 2),
        "misalignment_percent": round(misalignment_percent, 2)
    }


def get_visual_status(misalignment_percent):
    if misalignment_percent <= 5:
        return "PASS"
    elif misalignment_percent <= 15:
        return "WARNING"
    else:
        return "FAIL"


def calculate_visual_score(misalignment_percent):
    score = 100 - (misalignment_percent * 2)
    return max(0, round(score, 2))


# =========================================================
# VISUAL INSPECTION USING YOLO OR SIMULATION
# =========================================================
def class_matches(class_name, keywords):
    class_name = class_name.lower()

    for keyword in keywords:
        if keyword in class_name:
            return True

    return False


def extract_base_and_die_boxes(yolo_result):
    names = yolo_result.names
    boxes = yolo_result.boxes

    best_base = None
    best_die = None

    if boxes is None or len(boxes) == 0:
        return best_base, best_die

    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        if isinstance(names, dict):
            class_name = names.get(class_id, str(class_id))
        else:
            class_name = names[class_id]

        class_name = str(class_name)

        if class_matches(class_name, BASE_CLASS_KEYWORDS):
            if best_base is None or confidence > best_base["confidence"]:
                best_base = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                }

        elif class_matches(class_name, DIE_CLASS_KEYWORDS):
            if best_die is None or confidence > best_die["confidence"]:
                best_die = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                }

    return best_base, best_die


def draw_detection_result(image_path, processed_image_path, base_bbox, die_bbox, visual_status, misalignment_percent):
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image for drawing. Copying original instead.")
        shutil.copy(image_path, processed_image_path)
        return

    if base_bbox is not None:
        x1, y1, x2, y2 = [int(value) for value in base_bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            image,
            "BASE",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

    if die_bbox is not None:
        x1, y1, x2, y2 = [int(value) for value in die_bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 3)
        cv2.putText(
            image,
            "DIE",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 180, 0),
            2
        )

    label = f"{visual_status} | Misalignment: {misalignment_percent:.2f}%"

    cv2.putText(
        image,
        label,
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    cv2.imwrite(processed_image_path, image)


def simulate_visual_detection(image_path, processed_image_path):
    print("Using simulated visual detection result.")

    image = cv2.imread(image_path)

    if image is None:
        image_width = CAMERA_WIDTH
        image_height = CAMERA_HEIGHT
    else:
        image_height, image_width = image.shape[:2]

    base_bbox = (
        int(image_width * 0.20),
        int(image_height * 0.20),
        int(image_width * 0.80),
        int(image_height * 0.80)
    )

    shift_x = random.randint(-35, 35)
    shift_y = random.randint(-25, 25)

    die_bbox = (
        int(image_width * 0.27) + shift_x,
        int(image_height * 0.28) + shift_y,
        int(image_width * 0.73) + shift_x,
        int(image_height * 0.72) + shift_y
    )

    alignment = calculate_alignment(base_bbox, die_bbox)
    visual_status = get_visual_status(alignment["misalignment_percent"])
    visual_score = calculate_visual_score(alignment["misalignment_percent"])

    draw_detection_result(
        image_path,
        processed_image_path,
        base_bbox,
        die_bbox,
        visual_status,
        alignment["misalignment_percent"]
    )

    return {
        "base_detected": "YES",
        "die_detected": "YES",
        "base_confidence": 1.0,
        "die_confidence": 1.0,
        "base_bbox": bbox_to_string(base_bbox),
        "die_bbox": bbox_to_string(die_bbox),
        "x_offset_px": alignment["x_offset_px"],
        "y_offset_px": alignment["y_offset_px"],
        "offset_distance_px": alignment["offset_distance_px"],
        "offset_percent": alignment["offset_percent"],
        "overlap_percent": alignment["overlap_percent"],
        "misalignment_percent": alignment["misalignment_percent"],
        "visual_status": visual_status,
        "visual_score": visual_score
    }


def run_visual_inspection(image_path, processed_image_path, model):
    if model is None:
        return simulate_visual_detection(image_path, processed_image_path)

    print("Running YOLO visual inspection...")

    results = model(image_path, verbose=False)
    yolo_result = results[0]

    base_result, die_result = extract_base_and_die_boxes(yolo_result)

    if base_result is None or die_result is None:
        print("Base or die was not detected properly.")

        base_bbox = base_result["bbox"] if base_result else None
        die_bbox = die_result["bbox"] if die_result else None

        draw_detection_result(
            image_path,
            processed_image_path,
            base_bbox,
            die_bbox,
            "FAIL",
            100
        )

        return {
            "base_detected": "YES" if base_result else "NO",
            "die_detected": "YES" if die_result else "NO",
            "base_confidence": round(base_result["confidence"], 4) if base_result else 0,
            "die_confidence": round(die_result["confidence"], 4) if die_result else 0,
            "base_bbox": bbox_to_string(base_bbox),
            "die_bbox": bbox_to_string(die_bbox),
            "x_offset_px": "",
            "y_offset_px": "",
            "offset_distance_px": "",
            "offset_percent": "",
            "overlap_percent": "",
            "misalignment_percent": 100,
            "visual_status": "FAIL",
            "visual_score": 0
        }

    base_bbox = base_result["bbox"]
    die_bbox = die_result["bbox"]

    alignment = calculate_alignment(base_bbox, die_bbox)
    visual_status = get_visual_status(alignment["misalignment_percent"])
    visual_score = calculate_visual_score(alignment["misalignment_percent"])

    draw_detection_result(
        image_path,
        processed_image_path,
        base_bbox,
        die_bbox,
        visual_status,
        alignment["misalignment_percent"]
    )

    return {
        "base_detected": "YES",
        "die_detected": "YES",
        "base_confidence": round(base_result["confidence"], 4),
        "die_confidence": round(die_result["confidence"], 4),
        "base_bbox": bbox_to_string(base_bbox),
        "die_bbox": bbox_to_string(die_bbox),
        "x_offset_px": alignment["x_offset_px"],
        "y_offset_px": alignment["y_offset_px"],
        "offset_distance_px": alignment["offset_distance_px"],
        "offset_percent": alignment["offset_percent"],
        "overlap_percent": alignment["overlap_percent"],
        "misalignment_percent": alignment["misalignment_percent"],
        "visual_status": visual_status,
        "visual_score": visual_score
    }


# =========================================================
# DC VOLTAGE TEST
# =========================================================
def read_voltage(hardware):
    print("Reading voltage from ADS1115 A2...")

    if not RUN_WITH_REAL_HARDWARE:
        return round(random.uniform(2.20, 2.80), 3)

    ads_channel = hardware["ads_channel"]

    if ads_channel is None:
        print("ADS1115 not available. Simulating voltage reading.")
        return round(random.uniform(2.20, 2.80), 3)

    readings = []

    for _ in range(ADS_SAMPLE_COUNT):
        readings.append(ads_channel.voltage)
        time.sleep(ADS_SAMPLE_DELAY)

    voltage_value = sum(readings) / len(readings)
    return round(voltage_value, 3)


def get_electrical_status(voltage_value):
    if 2.30 <= voltage_value <= 2.70:
        return "PASS"

    if 2.20 <= voltage_value < 2.30:
        return "WARNING"

    if 2.70 < voltage_value <= 2.80:
        return "WARNING"

    return "FAIL"


def run_dc_voltage_test(hardware):
    lower_dc_platform(hardware)
    time.sleep(DC_STABILIZE_TIME)

    voltage_value = read_voltage(hardware)
    electrical_status = get_electrical_status(voltage_value)

    raise_dc_platform(hardware)

    print(f"Voltage: {voltage_value} V | Electrical Status: {electrical_status}")

    return voltage_value, electrical_status


# =========================================================
# FINAL RESULT LOGIC
# =========================================================
def calculate_final_die_score(visual_score, electrical_status):
    electrical_status = str(electrical_status).upper()

    final_score = float(visual_score)

    if electrical_status == "WARNING":
        final_score -= 10

    elif electrical_status in ["FAIL", "DEFECT"]:
        final_score -= 30

    return max(0, round(final_score, 2))


def get_tray_risk(visual_status, electrical_status, final_die_score):
    if visual_status == "FAIL" or electrical_status == "FAIL" or final_die_score < 70:
        return "HIGH"

    if visual_status == "WARNING" or final_die_score < 85:
        return "MEDIUM"

    return "LOW"


def get_final_sort_status(visual_status, electrical_status):
    visual_status = str(visual_status).upper()
    electrical_status = str(electrical_status).upper()

    if visual_status == "PASS" and electrical_status == "PASS":
        return "GOOD"

    if visual_status == "WARNING" and electrical_status in ["PASS", "WARNING"]:
        return "WARNING"

    if electrical_status in ["FAIL", "DEFECT"]:
        return "DEFECT"

    if visual_status in ["FAIL", "DEFECT"]:
        return "DEFECT"

    return "DEFECT"


# =========================================================
# QR AND CLOUD UPLOAD
# =========================================================
def generate_qr_link(tray_id):
    if is_placeholder(DASHBOARD_BASE_URL):
        return ""

    return f"{DASHBOARD_BASE_URL}?tray_id={tray_id}"


def generate_qr_image(tray_id, qr_link):
    if not qr_link:
        return ""

    qr_path = os.path.join(QR_DIR, f"{tray_id}_qr.png")

    qr_img = qrcode.make(qr_link)
    qr_img.save(qr_path)

    return qr_path


def upload_to_cloud(row, processed_image_path, qr_image_path):
    """
    Optional upload function.

    It sends:
    - inspection data
    - processed image as base64
    - QR image as base64, if available

    Your Google Apps Script can later:
    - append row into Google Sheets
    - upload processed image into Google Drive
    - return image_url if needed
    """

    if is_placeholder(GOOGLE_SCRIPT_URL):
        print("Google Apps Script URL not set. Skipping cloud upload.")
        return row

    payload = dict(row)
    payload["image_base64"] = file_to_base64(processed_image_path)
    payload["qr_image_base64"] = file_to_base64(qr_image_path)

    try:
        response = requests.post(GOOGLE_SCRIPT_URL, json=payload, timeout=20)
        print("Cloud response:", response.text)

        try:
            response_data = response.json()

            if "image_url" in response_data:
                row["image_url"] = response_data["image_url"]

        except Exception:
            pass

    except Exception as error:
        print("Cloud upload failed:", error)

    return row


# =========================================================
# MAIN INSPECTION SEQUENCE
# =========================================================
def run_single_inspection(hardware, model, tray_counter, die_counter):
    tray_id = f"T{tray_counter:03d}"
    die_id = f"D{die_counter:03d}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    image_name = f"{tray_id}_{die_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = os.path.join(CAPTURES_DIR, image_name)
    processed_image_path = os.path.join(PROCESSED_DIR, image_name)

    print("\n=================================================")
    print(f"Starting inspection for {tray_id} {die_id}")
    print("=================================================")

    # Station 1: visual inspection
    motor_forward(hardware)

    detected_visual = wait_for_ir_detection(hardware, "ir_visual")

    motor_stop(hardware)

    if not detected_visual:
        raise RuntimeError("Visual station IR sensor did not detect test subject.")

    capture_image(image_path)

    visual_result = run_visual_inspection(
        image_path=image_path,
        processed_image_path=processed_image_path,
        model=model
    )

    # Station 2: DC voltage test
    motor_forward(hardware)

    detected_dc = wait_for_ir_detection(hardware, "ir_dc")

    motor_stop(hardware)

    if not detected_dc:
        raise RuntimeError("DC test station IR sensor did not detect test subject.")

    voltage_value, electrical_status = run_dc_voltage_test(hardware)

    # Final result calculation
    final_die_score = calculate_final_die_score(
        visual_score=visual_result["visual_score"],
        electrical_status=electrical_status
    )

    tray_risk = get_tray_risk(
        visual_status=visual_result["visual_status"],
        electrical_status=electrical_status,
        final_die_score=final_die_score
    )

    final_sort_status = get_final_sort_status(
        visual_status=visual_result["visual_status"],
        electrical_status=electrical_status
    )

    qr_code_url = generate_qr_link(tray_id)
    qr_image_path = generate_qr_image(tray_id, qr_code_url)

    row = {
        "tray_id": tray_id,
        "die_id": die_id,
        "image_name": image_name,

        "base_detected": visual_result["base_detected"],
        "die_detected": visual_result["die_detected"],
        "base_confidence": visual_result["base_confidence"],
        "die_confidence": visual_result["die_confidence"],
        "base_bbox": visual_result["base_bbox"],
        "die_bbox": visual_result["die_bbox"],

        "x_offset_px": visual_result["x_offset_px"],
        "y_offset_px": visual_result["y_offset_px"],
        "offset_distance_px": visual_result["offset_distance_px"],
        "offset_percent": visual_result["offset_percent"],
        "overlap_percent": visual_result["overlap_percent"],
        "misalignment_percent": visual_result["misalignment_percent"],

        "visual_status": visual_result["visual_status"],
        "visual_score": visual_result["visual_score"],

        "voltage_value": voltage_value,
        "electrical_status": electrical_status,

        "final_die_score": final_die_score,
        "tray_risk": tray_risk,
        "timestamp": timestamp,

        "image_url": "",
        "qr_code_url": qr_code_url
    }

    # Optional cloud upload
    row = upload_to_cloud(row, processed_image_path, qr_image_path)

    # Save into local CSV
    append_row_to_csv(row)

    print("\nInspection saved.")
    print(f"Tray ID: {tray_id}")
    print(f"Die ID: {die_id}")
    print(f"Visual Status: {row['visual_status']}")
    print(f"Misalignment: {row['misalignment_percent']}%")
    print(f"Voltage: {row['voltage_value']} V")
    print(f"Electrical Status: {row['electrical_status']}")
    print(f"Final Die Score: {row['final_die_score']}")
    print(f"Tray Risk: {row['tray_risk']}")

    # Move sample to sorting area and sort
    move_motor_for_seconds(hardware, SORTING_TRAVEL_TIME)
    sort_sample(hardware, final_sort_status)

    return row


# =========================================================
# CLEANUP
# =========================================================
def cleanup(hardware):
    print("\nCleaning up system...")

    try:
        motor_stop(hardware)
        reset_all_servos(hardware)

    except Exception as error:
        print("Cleanup warning:", error)

    print("System stopped safely.")


# =========================================================
# MAIN PROGRAM
# =========================================================
def main():
    ensure_project_folders()
    create_csv_if_missing()

    hardware = setup_hardware()
    model = load_yolo_model()

    tray_counter, die_counter = load_counter()

    print("\nS.P.A.R.K.S system is ready.")
    print("Press ENTER to inspect one test subject.")
    print("Type q then ENTER to quit.\n")

    try:
        while True:
            user_input = input("Start inspection? ")

            if user_input.lower() == "q":
                break

            try:
                run_single_inspection(
                    hardware=hardware,
                    model=model,
                    tray_counter=tray_counter,
                    die_counter=die_counter
                )

                die_counter += 1

                if die_counter > 10:
                    die_counter = 1
                    tray_counter += 1

                save_counter(tray_counter, die_counter)

            except Exception as error:
                print("\nInspection error:", error)
                motor_stop(hardware)
                reset_all_servos(hardware)

    except KeyboardInterrupt:
        print("\nCTRL+C detected.")

    finally:
        cleanup(hardware)


if __name__ == "__main__":
    main()