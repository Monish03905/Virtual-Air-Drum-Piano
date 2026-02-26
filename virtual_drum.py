import importlib
import math
import os
import struct
import subprocess
import sys
import time
import urllib.request
import wave
from typing import Dict, List, Optional, Tuple

REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "pygame": "pygame",
}

def ensure_dependencies() -> None:
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"Installing missing dependency: {package_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_dependencies()

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame

# --- Configurations ---
SAMPLE_RATE = 44100
SOUND_FOLDER = os.path.join(os.path.dirname(__file__), "sounds")
MODEL_PATH = "hand_landmarker.task"

FINGER_TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
TAP_THRESHOLD = 0.045
COOLDOWN = 0.25

# --- Sound Plans ---
DRUM_LANES = ["Kick", "Snare", "HiHat", "Tom", "Clap"]
SOUND_PLAN_DRUMS = {
    "Kick": 180, "Snare": 320, "HiHat": 400, "Tom": 500, "Clap": 620,
    "Crash": 1000 # Hidden sound for flat-hand drum strikes
}

PIANO_LANES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
PIANO_SCALE = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]
SOUND_PLAN_PIANO = {
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, 
    "G4": 392.00, "A4": 440.00, "B4": 493.88, "C5": 523.25,
    "D5": 587.33, "E5": 659.25, "F5": 698.46, "G5": 783.99
}

# --- Helper Functions ---
def ensure_model_exists() -> None:
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe Hand Landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

def ensure_sound_folder() -> None:
    os.makedirs(SOUND_FOLDER, exist_ok=True)

def generate_sine_wave(path: str, frequency: float, instr_type: str) -> None:
    duration = 0.35 if instr_type == "drum" else 1.0 
    sample_count = int(SAMPLE_RATE * duration)
    amplitude = 32767
    frames = []
    
    for n in range(sample_count):
        t = n / SAMPLE_RATE
        if instr_type == "drum":
            decay = math.exp(-12.0 * t) 
        else:
            decay = math.exp(-3.0 * t)   
            
        value = int(amplitude * 0.5 * decay * math.sin(2 * math.pi * frequency * t))
        frames.append(struct.pack('<h', value))
        
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"".join(frames))

def prepare_sound_assets() -> Dict[str, Dict[str, str]]:
    ensure_sound_folder()
    assets = {"drum": {}, "piano": {}}
    
    for label, frequency in SOUND_PLAN_DRUMS.items():
        path = os.path.join(SOUND_FOLDER, f"drum_{label}.wav")
        if not os.path.exists(path): generate_sine_wave(path, frequency, "drum")
        assets["drum"][label] = path
        
    for label, frequency in SOUND_PLAN_PIANO.items():
        path = os.path.join(SOUND_FOLDER, f"piano_{label}.wav")
        if not os.path.exists(path): generate_sine_wave(path, frequency, "piano")
        assets["piano"][label] = path
        
    return assets

FINGER_COLORS = {
    "thumb": (255, 100, 100), "index": (255, 210, 30),
    "middle": (80, 220, 120), "ring": (100, 150, 255), "pinky": (200, 120, 255)
}

def init_audio(sound_files: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, pygame.mixer.Sound]]:
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
    pygame.init()
    pygame.mixer.set_num_channels(32) # Allow many sounds to play at once for chords
    loaded_sounds = {"drum": {}, "piano": {}}
    
    for mode in ["drum", "piano"]:
        for zone, path in sound_files[mode].items():
            sound = pygame.mixer.Sound(path)
            
            # --- VOLUME NORMALIZER ADDED HERE ---
            # Lowering the individual sound volumes prevents audio clipping
            if mode == "piano":
                sound.set_volume(0.3)  # 3 notes played at once will equal ~90% total volume
            elif mode == "drum":
                sound.set_volume(0.5)  # Crash + Kick together will equal 100% total volume
                
            loaded_sounds[mode][zone] = sound
            
    return loaded_sounds

def is_flat_hand(hand_landmarks) -> bool:
    """Checks if the hand is flat by calculating if at least 3 fingers are fully extended."""
    extended_fingers = 0
    wrist = hand_landmarks[0]
    
    for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip = hand_landmarks[tip_id]
        pip = hand_landmarks[pip_id]
        dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        dist_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        
        # If tip is significantly further from the wrist than the middle knuckle, it's extended
        if dist_tip > dist_pip * 1.25:
            extended_fingers += 1
            
    return extended_fingers >= 3

# --- UI Drawing ---
def draw_drum_art(frame, x1, x2, y1, y2, name: str) -> None:
    color = (200, 200, 200)
    center_x = (x1 + x2) // 2
    center_y = y1 + (y2 - y1) // 2
    cv2.circle(frame, (center_x, center_y), 40, color, 3)
    cv2.putText(frame, name, (center_x - 20, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_zones(frame, width: int, height: int, mode: str) -> List[Tuple[int, int, int, int, str]]:
    overlay = frame.copy()
    zones: List[Tuple[int, int, int, int, str]] = []
    zone_names = DRUM_LANES if mode == "drum" else PIANO_LANES
    zone_width = width // len(zone_names)

    for index, name in enumerate(zone_names):
        x1 = index * zone_width
        x2 = x1 + zone_width
        
        if mode == "drum":
            cv2.rectangle(overlay, (x1, 0), (x2, height), (50, 50, 50), -1)
        else:
            cv2.rectangle(overlay, (x1, 0), (x2, height), (255, 255, 255), -1)
            cv2.rectangle(overlay, (x1, 0), (x2, height), (0, 0, 0), 2)
            cv2.putText(overlay, name, (x1 + 10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            
        zones.append((x1, x2, 0, height, name))

    alpha = 0.15 if mode == "drum" else 0.25
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    if mode == "drum":
        for x1, x2, y1, y2, name in zones:
            draw_drum_art(frame, x1, x2, y1, y2, name)
            
    return zones

def find_zone(zones, x_pixel: int) -> Optional[str]:
    for x1, x2, _, _, name in zones:
        if x1 <= x_pixel <= x2:
            return name
    return None

# --- Main Logic ---
def main() -> None:
    ensure_model_exists()
    sound_files = prepare_sound_assets()
    sounds = init_audio(sound_files)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_finger_y: Dict[str, float] = {finger: 0.0 for finger in FINGER_TIP_IDS}
    last_trigger_time: Dict[str, float] = {finger: 0.0 for finger in FINGER_TIP_IDS}
    
    prev_palm_y = 0.0
    last_palm_trigger = 0.0
    
    active_visuals: List[Tuple[int, int, Tuple[int, int, int], float, str]] = []
    current_mode = "piano" 

    try:
        while True:
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            zones = draw_zones(frame, width, height, current_mode)

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                
                # --- Chord / Flat Hand Logic ---
                palm_landmark = hand_landmarks[9] # Base of the middle finger
                palm_x = int(palm_landmark.x * width)
                palm_y = int(palm_landmark.y * height)
                
                delta_palm = palm_landmark.y - prev_palm_y
                prev_palm_y = palm_landmark.y
                recent_palm = time.time() - last_palm_trigger
                
                chord_triggered = False
                
                if is_flat_hand(hand_landmarks) and 0.3 > delta_palm > TAP_THRESHOLD and recent_palm > COOLDOWN:
                    zone_name = find_zone(zones, palm_x)
                    if zone_name:
                        if current_mode == "piano":
                            # Play the triad chord
                            base_idx = PIANO_SCALE.index(zone_name)
                            chord_notes = [PIANO_SCALE[base_idx], PIANO_SCALE[base_idx + 2], PIANO_SCALE[base_idx + 4]]
                            for note in chord_notes:
                                sounds["piano"][note].play()
                        elif current_mode == "drum":
                            # Play Crash Cymbal + Kick Drum
                            sounds["drum"]["Crash"].play()
                            sounds["drum"]["Kick"].play()
                            
                        last_palm_trigger = time.time()
                        chord_triggered = True
                        
                        # Suppress individual finger taps to prevent chaotic noise
                        for finger in FINGER_TIP_IDS:
                            last_trigger_time[finger] = time.time()
                            
                        active_visuals.append((palm_x, palm_y, (255, 255, 255), time.time(), "chord"))
                
                # --- Individual Finger Logic ---
                if not chord_triggered:
                    for finger, tip_id in FINGER_TIP_IDS.items():
                        landmark = hand_landmarks[tip_id]
                        x_pixel = int(landmark.x * width)
                        y_pixel = int(landmark.y * height)

                        zone_name = find_zone(zones, x_pixel)
                        if zone_name:
                            cv2.circle(frame, (x_pixel, y_pixel), 10, FINGER_COLORS[finger], -1)

                        delta_y = landmark.y - prev_finger_y[finger]
                        prev_finger_y[finger] = landmark.y
                        recent = time.time() - last_trigger_time[finger]
                        
                        if zone_name and 0.3 > delta_y > TAP_THRESHOLD and recent > COOLDOWN:
                            sounds[current_mode][zone_name].play() 
                            last_trigger_time[finger] = time.time()
                            active_visuals.append((x_pixel, y_pixel, FINGER_COLORS[finger], time.time(), "finger"))
            else:
                # Reset tracking variables when hand leaves frame
                for finger in FINGER_TIP_IDS:
                    prev_finger_y[finger] = 0.0
                prev_palm_y = 0.0

            # --- Tap Animations ---
            now = time.time()
            active_visuals = [v for v in active_visuals if now - v[3] < 0.3]
            for x, y, color, start_time, vis_type in active_visuals:
                progress = (now - start_time) / 0.3
                if vis_type == "finger":
                    radius = int(30 * (1 - progress)) + 1
                    cv2.circle(frame, (x, y), radius, color, 2)
                elif vis_type == "chord":
                    # Larger, hollow shockwave for chords
                    radius = int(100 * (1 - progress)) + 1
                    cv2.circle(frame, (x, y), radius, (200, 255, 200), 4)

            # Instructions
            cv2.putText(frame, "Press 'M' to switch modes | Press 'Q' to quit", (20, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            mode_text = f"Current Mode: {current_mode.upper()}"
            cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Virtual Studio", frame)
            
            # Keyboard Inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'): current_mode = "piano" if current_mode == "drum" else "drum"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()