from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.core.validators import validate_email
from django.core.exceptions import ValidationError

import cv2
from ultralytics import YOLO
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import csv
from time import time

# ---------------- GLOBAL STATE ---------------- #

model = None

violation_log = []  

violation_data = {
    "last_violation_frame": None,
    "last_violation_time": None,
    "violation_count": 0
}

VIOLATION_DELAY = 7     # seconds
COOLDOWN_TIME = 5       # seconds

violation_timer = {
    "start_time": None,
    "active": False,
    "cooldown": False
}

violation_confirmed = False

# ---------------- AUTH ---------------- #

def login_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        user = authenticate(
            request,
            username=request.POST.get("username"),
            password=request.POST.get("password")
        )
        if user:
            auth_login(request, user)
            return redirect("home")
        return render(request, "detection/login.html", {"error": "Invalid credentials"})

    return render(request, "detection/login.html")


def logout_view(request):
    auth_logout(request)
    return redirect("login")

def signup_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        password2 = request.POST.get("password2")

        if password != password2:
            return render(request, "detection/signup.html", {
                "error": "Passwords do not match"
            })

        if User.objects.filter(username=username).exists():
            return render(request, "detection/signup.html", {
                "error": "Username already exists"
            })

        if User.objects.filter(email=email).exists():
            return render(request, "detection/signup.html", {
                "error": "Email already registered"
            })

        try:
            validate_email(email)
        except ValidationError:
            return render(request, "detection/signup.html", {
                "error": "Invalid email address"
            })

        User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )

        return redirect("login")

    return render(request, "detection/signup.html")


# ---------------- PAGES ---------------- #

@login_required
def home(request):
    return render(request, "detection/home.html", {"username": request.user.username})


@login_required
def mode2(request):
    return render(request, "detection/mode2.html", {"username": request.user.username})


@login_required
def view_violations(request):
    return render(request, "detection/violations.html", {
        "violations": violation_log,
        "username": request.user.username
    })


# ---------------- MODEL ---------------- #

def load_model():
    global model
    if model is None:
        # Load your custom PPE model from trained_models folder
        model_path = "trained_models/best.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✅ Loaded custom PPE model from {model_path}")
            print(f"📋 Model classes: {model.names}")
        else:
            print(f"❌ ERROR: {model_path} not found!")
            model = YOLO("yolov8n.pt")
            print("⚠️ WARNING: Using base model - PPE detection will not work!")


# ---------------- VIDEO STREAM ---------------- #

def generate_frames():
    global model, violation_timer, violation_confirmed

    load_model()
    camera = cv2.VideoCapture(0)

    # Extended PPE class names (for custom models)
    helmet_classes = {
        "helmet", "hardhat", "safety-helmet", "hard-hat",
        "hard hat", "safety helmet", "white helmet", "yellow helmet"
    }
    vest_classes = {
        "vest", "safety-vest", "hi-vis", "safety vest",
        "high-visibility vest", "reflective vest", "hiviz"
    }
    
    # All possible PPE classes
    all_ppe_classes = helmet_classes | vest_classes | {
        "gloves", "boots", "goggles", "mask", "safety-goggles",
        "safety boots", "safety gloves"
    }

    while True:
        success, frame = camera.read()
        if not success:
            break

        current_time = time()
        
        # Store all detected objects with their boxes
        detected_helmets = []
        detected_vests = []
        detected_persons = []
        detected_other_ppe = []

        # Run detection
        results = model(frame, conf=0.4, verbose=False)

        # First pass: collect all detections
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls].lower()

                # Categorize detections with stricter confidence for vests
                # IMPORTANT: Check for negative classes first (no-helmet, no-vest)
                if "no-helmet" in name or "no-hardhat" in name or "nohelmet" in name:
                    # This is a person WITHOUT helmet - skip this detection
                    continue
                elif "no-vest" in name or "novest" in name or "no-hi-vis" in name:
                    # This is a person WITHOUT vest - skip this detection
                    continue
                elif name in helmet_classes or "helmet" in name or "hardhat" in name or "hard-hat" in name:
                    if conf >= 0.4:  # Helmet detection threshold
                        detected_helmets.append((x1, y1, x2, y2, name, conf))
                elif name in vest_classes or "vest" in name or "hi-vis" in name or "hiviz" in name:
                    if conf >= 0.65:  # Higher threshold for vest to reduce false positives
                        detected_vests.append((x1, y1, x2, y2, name, conf))
                elif name == "person":
                    detected_persons.append((x1, y1, x2, y2))
                elif name in all_ppe_classes:
                    detected_other_ppe.append((x1, y1, x2, y2, name, conf))

        # Helper function to check overlap with tolerance
        def has_overlap(boxA, boxB, tolerance=50):
            ax1, ay1, ax2, ay2 = boxA
            bx1, by1, bx2, by2 = boxB
            
            # Expand person box slightly to catch nearby PPE
            ax1 -= tolerance
            ay1 -= tolerance
            ax2 += tolerance
            ay2 += tolerance
            
            return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

        # Draw detected PPE items (always green)
        for x1, y1, x2, y2, name, conf in detected_helmets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for x1, y1, x2, y2, name, conf in detected_vests:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for x1, y1, x2, y2, name, conf in detected_other_ppe:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)
            label = f"{name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        # Process each person
        person_has_violation = False
        
        for person_box in detected_persons:
            px1, py1, px2, py2 = person_box
            
            # Check if this person has both helmet AND vest with more tolerance
            has_helmet = any(has_overlap(person_box, (hx1, hy1, hx2, hy2), tolerance=80) 
                           for hx1, hy1, hx2, hy2, _, _ in detected_helmets)
            has_vest = any(has_overlap(person_box, (vx1, vy1, vx2, vy2), tolerance=60) 
                          for vx1, vy1, vx2, vy2, _, _ in detected_vests)

            # Person must have BOTH helmet and vest to be compliant
            if has_helmet and has_vest:
                color = (0, 255, 0)
                label = "PPE OK"
            elif has_helmet and not has_vest:
                color = (0, 0, 255)
                label = "NO VEST"
                person_has_violation = True
            elif has_vest and not has_helmet:
                color = (0, 0, 255)
                label = "NO HELMET"
                person_has_violation = True
            else:
                color = (0, 0, 255)
                label = "NO PPE"
                person_has_violation = True

            # Draw person box with thicker line
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)
            
            # Draw label with background for better visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (px1, py1 - label_size[1] - 10), 
                         (px1 + label_size[0], py1), color, -1)
            cv2.putText(frame, label, (px1, py1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Handle violation timer logic
        if person_has_violation:
            # Start timer if not already active and not in cooldown
            if not violation_timer["active"] and not violation_timer["cooldown"]:
                violation_timer["active"] = True
                violation_timer["start_time"] = current_time
                violation_confirmed = False

            # Check if violation delay has passed
            if violation_timer["active"]:
                elapsed = current_time - violation_timer["start_time"]
                
                # Show countdown
                remaining = int(VIOLATION_DELAY - elapsed)
                if remaining > 0:
                    cv2.putText(frame, f"Warning: {remaining}s", (40, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                
                if elapsed >= VIOLATION_DELAY:
                    timestamp = datetime.now()

                    violation_confirmed = True
                    violation_timer["active"] = False
                    violation_timer["cooldown"] = True
                    violation_timer["start_time"] = current_time

                    # SAVE IMAGE
                    os.makedirs("media/violations", exist_ok=True)
                    filename = f"violation_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                    path = f"media/violations/{filename}"
                    cv2.imwrite(path, frame)

                    violation_log.append({
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "violation_type": "Person without PPE",
                        "image_path": path
                    })

                    violation_data["violation_count"] += 1
                    violation_data["last_violation_frame"] = frame.copy()
                    violation_data["last_violation_time"] = timestamp
        else:
            # No violation detected - reset timer
            violation_timer["active"] = False
            if not violation_timer["cooldown"]:
                violation_confirmed = False

        # Handle cooldown
        if violation_timer["cooldown"]:
            if current_time - violation_timer["start_time"] >= COOLDOWN_TIME:
                violation_timer["cooldown"] = False
                violation_confirmed = False

        # Display violation warning
        if violation_confirmed:
            cv2.putText(frame, "PPE VIOLATION CONFIRMED",
                       (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Display detection info
        info_text = f"Helmets: {len(detected_helmets)} | Vests: {len(detected_vests)} | Persons: {len(detected_persons)}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    camera.release()


@login_required
def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------- DOWNLOADS ---------------- #

@login_required
@csrf_exempt
def download_receipt(request):
    global violation_data

    if violation_data['last_violation_frame'] is None:
        return HttpResponse("No violation detected yet", status=404)

    frame = violation_data['last_violation_frame']
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    receipt_width = 800
    receipt_height = pil_image.height + 280
    receipt = Image.new("RGB", (receipt_width, receipt_height), "white")

    frame_width = 760
    frame_height = int(pil_image.height * (frame_width / pil_image.width))
    pil_image = pil_image.resize((frame_width, frame_height))
    receipt.paste(pil_image, (20, 20))

    draw = ImageDraw.Draw(receipt)

    try:
        title_font = ImageFont.truetype("arial.ttf", 30)
        text_font = ImageFont.truetype("arial.ttf", 20)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    y = frame_height + 40

    timestamp = violation_data['last_violation_time']
    if timestamp is None:
        timestamp = datetime.now()

    draw.text((20, y), "⚠ PPE VIOLATION REPORT", fill="red", font=title_font)
    y += 40

    draw.text((20, y), "Violation Type: PPE Not Worn", fill="black", font=text_font)
    y += 30

    draw.text((20, y), f"Date & Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
             fill="black", font=text_font)
    y += 30

    draw.text((20, y), f"Reported By: {request.user.username}",
             fill="black", font=text_font)
    y += 30

    draw.text((20, y), f"Total Violations Today: {violation_data['violation_count']}",
             fill="black", font=text_font)

    img_io = io.BytesIO()
    receipt.save(img_io, "JPEG", quality=95)
    img_io.seek(0)

    response = HttpResponse(img_io, content_type="image/jpeg")
    response["Content-Disposition"] = (
        f'attachment; filename="PPE_Violation_{timestamp.strftime("%Y%m%d_%H%M%S")}.jpg"'
    )

    return response


@login_required
def download_excel(request):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=ppe_violation_log.csv"

    writer = csv.writer(response)
    writer.writerow(["#", "Timestamp", "Violation", "Image"])

    for i, v in enumerate(violation_log, 1):
        writer.writerow([i, v["timestamp"], v["violation_type"], v["image_path"]])

    return response


@login_required
def get_violation_status(request):
    return JsonResponse({
        "has_violation": violation_data["last_violation_frame"] is not None,
        "violation_count": violation_data["violation_count"],
        "total_logged": len(violation_log)
    })