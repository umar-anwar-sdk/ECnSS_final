import os
import cv2
from datetime import datetime, timedelta
import time
from django.shortcuts import render, redirect
import numpy as np
from django.utils import timezone
from ultralytics import YOLO
from .sort import Sort
from .utils import paddle_ocr
from django.core.files.base import ContentFile
from .models import detectionRecord, CameraConfig
import threading
from django.http import HttpResponse
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

channel_layer = get_channel_layer()

global_stop_signal = threading.Event()
stop_signal = threading.Event() 
active_threads = [] 
def stop():
    global global_stop_signal, stop_signal, active_threads
    print("Not Set : ", global_stop_signal)
    global_stop_signal.set()
    stop_signal.set()
    print("Set : ", global_stop_signal)
    for thread in active_threads:
        thread.join(timeout=2.0)  
    active_threads.clear()
    print("All detection threads stopped")

def clear():
    global global_stop_signal, stop_signal, active_threads
    global_stop_signal.clear()
    stop_signal.clear()
    active_threads.clear()

def bike_detection(frame, license_plate,status,detection_type, cropedframe=None):
    try:
        model_path = "model/new_helmet_model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        bike_model = YOLO(model_path)
        classNames = ['With Helmet', 'Without Helmet']

        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")

        results = bike_model(frame, stream=True)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.xyxy.shape[0] == 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                if cls < len(classNames):
                    if license_plate and x1 < x2 and y1 < y2: 
                        license_plate.has_helmet = True if classNames[cls] == 'With Helmet' else False
                        license_plate.confidence = f"{conf * 100:.2f}%"
                        license_plate.detected_type = classNames[cls]  

                        detection_crop = frame[y1:y2, x1:x2]
                        if detection_crop.size > 0: 
                            _, buffer = cv2.imencode('.jpg', detection_crop)
                            license_plate.helmet_image.save(
                                f'helmet_detection_{license_plate.id}.jpg',
                                ContentFile(buffer.tobytes())
                            )
                            license_plate.save()
                            async_to_sync(channel_layer.group_send)(
                            "detection_group", {
                                "type": "send_detection",
                                "data": {
                                    "type": classNames[cls],
                                    "confidence": conf,
                                    "license_plate_id": license_plate.id,
                                    "status": status,
                                    "detection_type":detection_type
                                    
                                }
                            }
                            )
                        detections.append({
                            'type': classNames[cls],
                            'confidence': conf,
                        })

    except Exception as e:
        print(f"Error in bike detection: {e}")
        return frame, []

    return frame, detections
def vehicle_detection(frame, license_plate, status, detection_type, cropped_frame=None):
    print("Vehicle Detection in function")

    try:
        model_path = "model/vehicle.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        vehicle_model = YOLO(model_path)
        classNames = ['Distracted', 'Helmet', 'Drowsy', 'No Seatbelt', 'Seatbelt']

        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")

        results = vehicle_model(frame, stream=True)
        detections = []
        has_seatbelt_detected = None  # To track if seatbelt detection happened

        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.xyxy.shape[0] == 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                # Process seatbelt detections (class index 3 = No Seatbelt, 4 = Seatbelt)
                if cls in [3, 4]:
                    if x1 < x2 and y1 < y2:
                        has_seatbelt = (cls == 4)  # True for Seatbelt, False for No Seatbelt
                        has_seatbelt_detected = has_seatbelt

                        if license_plate:
                            license_plate.has_seatbelt = has_seatbelt
                            license_plate.confidence = f"{conf * 100:.2f}%"
                            print(f"License plate updated: Seatbelt={license_plate.has_seatbelt}")

                        # Save cropped seatbelt detection image
                        detection_crop = frame[y1:y2, x1:x2]
                        if detection_crop.size > 0:
                            _, buffer = cv2.imencode('.jpg', detection_crop)
                            license_plate.setbelt_image.save(
                                f'seatbelt_detection_{license_plate.id}.jpg',
                                ContentFile(buffer.tobytes())
                            )

                        # Save to database
                        license_plate.save()

                        # Send WebSocket message
                        async_to_sync(channel_layer.group_send)(
                            "detection_group",
                            {
                                "type": "send_detection",
                                "data": {
                                    "has_seatbelt": has_seatbelt,
                                    "confidence": conf,
                                    "license_plate_id": license_plate.id,
                                    "status": status,
                                    "detection_type": detection_type,
                                    "plate_number": license_plate.plate_number,
                                    "vehicle_image": license_plate.license_plate_image.url if license_plate.license_plate_image else None,
                                }
                            }
                        )
                        detections.append({
                            "has_seatbelt": has_seatbelt,
                            "confidence": conf,
                        })

        if has_seatbelt_detected is None:
            async_to_sync(channel_layer.group_send)(
                "detection_group",
                {
                    "type": "send_detection",
                    "data": {
                        "has_seatbelt": "N/A",
                        "confidence": "N/A",
                        "license_plate_id": license_plate.id,
                        "status": status,
                        "detection_type": detection_type,
                        "plate_number": license_plate.plate_number,
                        "vehicle_image": license_plate.license_plate_image.url if license_plate.license_plate_image else None,
                    }
                }
            )

    except Exception as e:
        print(f"Error in vehicle detection: {e}")
        return frame, []

    return frame, detections

def process_camera_stream(camera,stop_signal):
    try:
        camera_url = '/home/umar/Desktop/EC&SS/ECnSS/ECSS/test/22.mp4'
        # camera_url= "http://192.168.137.159:8080"
        # camera_url = camera.generated_url
        detection_type = camera.camera_type     
        while not stop_signal.is_set() and not global_stop_signal.is_set():
                cap = cv2.VideoCapture(camera_url)
                if not cap.isOpened():
                    raise FileNotFoundError(f"Cannot open stream: {camera_url}")
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 30  
                frame_interval = max(1, int(fps / 2))
                vehicle_model_path = "model/yolo11n.pt"
                license_plate_model_path = "model/number_plate.pt"    
                if not os.path.exists(vehicle_model_path):
                    raise FileNotFoundError(f"Vehicle model not found at {vehicle_model_path}")
                if not os.path.exists(license_plate_model_path):
                    raise FileNotFoundError(f"License plate model not found at {license_plate_model_path}")        
                vehicle_model = YOLO(vehicle_model_path)
                license_plate_model = YOLO(license_plate_model_path)
                tracker = Sort()
                CLASS_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 17: "bike"}
                frame_count = 0
                group_name = f"camera_{camera.id}"
                while not stop_signal.is_set() and not global_stop_signal.is_set():
                    if global_stop_signal.is_set():
                        print(f"Stopping detection for camera {camera.id} due to global stop signal")
                        break
                    if stop_signal.is_set():
                        print(f"Stopping detection for camera {camera.id} due to stop signal")
                        return  
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        stop_signal.set() 
                        break           
                    if frame_count % frame_interval != 0:
                        frame_count += 1
                        continue                
                    frame_count += 1         
                    try:
                        vehicle_results = vehicle_model(frame, conf=0.25) 
                        detections = []
                        for vehicle in vehicle_results[0].boxes:
                            if len(vehicle.xyxy) > 0:
                                x1, y1, x2, y2 = map(int, vehicle.xyxy[0])
                                confidence = float(vehicle.conf[0])
                                class_index = int(vehicle.cls[0])                       
                                if class_index in CLASS_MAP:
                                    class_name = CLASS_MAP[class_index]
                                    detections.append([x1, y1, x2, y2, confidence])                   
                        detections = np.array(detections) if detections else np.empty((0, 5))
                        annotated_frame = frame.copy()               
                        if detections.size > 0:
                            tracked_objects = tracker.update(detections)
                            for obj in tracked_objects:
                                x1, y1, x2, y2, obj_id = map(int, obj[:5])
                                obj_id = int(obj_id)
                                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                                    continue
                                vehicle_crop = frame[y1:y2, x1:x2]
                                if vehicle_crop.size == 0:
                                    continue
                                for vehicle in vehicle_results[0].boxes:
                                    if len(vehicle.xyxy) > 0:
                                        vx1, vy1, vx2, vy2 = map(int, vehicle.xyxy[0])
                                        if abs(vx1 - x1) < 10 and abs(vy1 - y1) < 10:  
                                            class_index = int(vehicle.cls[0])
                                            if class_index in CLASS_MAP:
                                                class_name = CLASS_MAP[class_index]                              
                                                plate_results = license_plate_model(vehicle_crop, conf=0.5)
                                                for plate in plate_results[0].boxes:
                                                    if plate.xyxy.ndim == 2 and plate.xyxy.shape[0] > 0:
                                                        px1, py1, px2, py2 = map(int, plate.xyxy[0])                                              
                                                        if px1 >= px2 or py1 >= py2:
                                                            continue                                             
                                                        plate_x1, plate_y1 = x1 + px1, y1 + py1
                                                        plate_x2, plate_y2 = x1 + px2, y1 + py2                                             
                                                        if plate_x1 >= plate_x2 or plate_y1 >= plate_y2 or \
                                                        plate_x1 < 0 or plate_y1 < 0 or \
                                                        plate_x2 > frame.shape[1] or plate_y2 > frame.shape[0]:
                                                            continue
                                                        plate_crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                                                        if plate_crop.size == 0:
                                                            continue   
                                                        plate_text = paddle_ocr(frame, plate_x1, plate_y1, plate_x2, plate_y2)
                                                    if plate_text:
                                                        plate_number = plate_text[0]
                                                        confidence_score = f'{int(plate_text[1] * 100)}%'
                                                        class_name = CLASS_MAP.get(class_index, "unknown")

                                                        last_record = detectionRecord.objects.filter(plate_number=plate_number).order_by('-check_in_time').first()
                                                        current_time = timezone.now()
                                                        status = "Unknown status"
                                                        license_plate = None

                                                        if detection_type == '1':  
                                                            if last_record and last_record.detection_type == '1' and last_record.check_out_time is None:
                                                                time_difference = (current_time - last_record.check_in_time).total_seconds() / 3600  
                                                                
                                                                if time_difference >= 12:
                                                                    last_record.check_out_miss = True 
                                                                    last_record.save()

                                                                    status = "Check-out missed ⏳ (New record created ✅)"
                                                                    license_plate = detectionRecord.objects.create(
                                                                        vehicle_id=obj_id,
                                                                        plate_number=plate_number,
                                                                        confidence=confidence_score,
                                                                        vehicle_class=class_name,
                                                                        detection_type='1',
                                                                        check_in_time=current_time,
                                                                        status=status
                                                                    )
                                                                else:
                                                                    status = "Ignored ❌ (Already checked in)"
                                                                    license_plate = last_record
                                                            else:
                                                                print("New record created")
                                                                status = "New record created ✅"
                                                                license_plate = detectionRecord.objects.create(
                                                                    vehicle_id=obj_id,
                                                                    plate_number=plate_number,
                                                                    confidence=confidence_score,
                                                                    vehicle_class=class_name,
                                                                    detection_type='1',
                                                                    check_in_time=current_time,
                                                                    status=status
                                                                )                                                       
                                                        elif detection_type == '0':  
                                                            if last_record and last_record.detection_type == '1' and last_record.check_out_time is None:
                                                                last_record.detection_type = '0'
                                                                last_record.check_out_time = current_time
                                                                last_record.status = f"Checked out ✅ ({plate_number})"
                                                                last_record.save()
                                                                status = f"Checked out ✅ ({plate_number})"
                                                                license_plate = last_record  
                                                            else:
                                                                status = "Ignored ❌ (No active check-in found)"
                                                                license_plate = last_record  
                                                        websocket_data = {
                                                            "type": "send_detection",
                                                            "data": {
                                                                "vehicle_id": obj_id,
                                                                "plate_number": plate_number,
                                                                "confidence": confidence_score,
                                                                "vehicle_class": class_name,
                                                                "status": status,
                                                                "detection_type": detection_type,
                                                                "check_in_time": current_time.isoformat() if detection_type == '1' else (last_record.check_in_time.isoformat() if last_record else None),
                                                                "check_out_time": current_time.isoformat() if detection_type == '0' else None,
                                                                "license_plate_id": license_plate.id if license_plate else None
                                                            }
                                                        }
                                                                                                           
                                                        async_to_sync(channel_layer.group_send)(
                                                            group_name,
                                                            websocket_data
                                                        )
                                                        print("Sent data to websocket")                                                  
                                                        if license_plate:
                                                            _, buffer = cv2.imencode('.jpg', plate_crop)
                                                            license_plate.license_plate_image.save(
                                                                f'plate_{license_plate.id}.jpg',
                                                                ContentFile(buffer.tobytes())
                                                            )                                                          
                                                            _, buffer = cv2.imencode('.jpg', vehicle_crop)
                                                            license_plate.vehicle_image.save(
                                                                f'vehicle_{license_plate.id}.jpg',
                                                                ContentFile(buffer.tobytes())
                                                            )
                                                            _, buffer = cv2.imencode('.jpg', frame)
                                                            license_plate.full_frame_image.save(
                                                                f'full_frame_{license_plate.id}.jpg',
                                                                ContentFile(buffer.tobytes())
                                                                )
                                                            print(class_name)                                                           
                                                            if class_name.lower() in ["car", "truck", "bus"]:
                                                                print("Vehicle Detection")
                                                                vehicle_detection(vehicle_crop, license_plate, status, detection_type, annotated_frame)
                            
                                                            elif class_name.lower() in ["motorcycle", "bike"]:
                                                                bike_detection(frame, license_plate, status, detection_type, annotated_frame)


                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"Detection error for stream: {e}")
                cap.release()
                cv2.destroyAllWindows()
                
    except Exception as e:
        print(f"Error in processing camera stream: {e}")

def start_camera_detections(request):
    global active_threads
    stop_all_camera_detections()
    cameras = CameraConfig.objects.all()
    for camera in cameras:
        stop_signal=threading.Event()
        thread = threading.Thread(target=process_camera_stream, args=(camera,stop_signal))
        thread.daemon = True  
        thread.start()
        active_threads.append((thread,stop_signal))
        time.sleep(2)
    return True, "Detection started successfully!"

def stop_all_camera_detections():
    global active_threads
    for _, stop_signal in active_threads:
        stop_signal.set()
    for thread, _ in active_threads:
        thread.join(timeout=5) 
    active_threads.clear()
    print("All threads stopped.")
    return True, "Detection Stop successfully!"

