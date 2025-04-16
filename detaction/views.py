from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.contrib import messages
from .models import detectionRecord,CameraConfig,DETECTION_CHOICES
from .detection_utils import start_camera_detections,stop_all_camera_detections
from django.shortcuts import render,get_object_or_404
import threading
from datetime import datetime
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
global_stop_signal = threading.Event()
stop_signal=threading.Event()
active_threads = []
@login_required(login_url='login')
def home(request):
    cameras = CameraConfig.objects.all()
    context = {
        'cameras': cameras,
    }
    return render(request, 'index.html', context)

def stop():
    global global_stop_signal
    print("Not Set : ",global_stop_signal)
    global_stop_signal.set()
    print("Set : ",global_stop_signal)
def clear():
    global global_stop_signal
    global_stop_signal.clear()

def stop_detection(request):
    try:
        success, message = stop_all_camera_detections()
        return JsonResponse({"success": success, "message": message})
    except Exception as e:
        return JsonResponse({"success": False, "message": f"Error: {str(e)}"})

def start_detections_view(request):
    try:
        success, message = start_camera_detections(request)
        return JsonResponse({"success": success, "message": message})
    except Exception as e:
        return JsonResponse({"success": False, "message": f"Error: {str(e)}"})


@login_required(login_url='login')
def generate_camera_url(request):
    if request.method == 'POST':
        Username = request.POST.get("username")
        Password = request.POST.get("password")
        ip = request.POST.get("ip")
        channelName = request.POST.get("channel_name")
        checkOption = request.POST.get("check_option")
            
        camera_config = CameraConfig(
            username=Username,
            password=Password,
            ip=ip,
            channel_name=channelName,
            camera_type=int(checkOption)
        )
        
        camera_config.save()
        return redirect('home')
    else:
        return render(request, 'camera_form.html')
@login_required(login_url='login')
def Camera_list (request):
    cameras = CameraConfig.objects.all()
    context = {
        'cameras': cameras,
    }
    return render(request, 'camera_list.html', context)
@login_required(login_url='login')
def camera_delete(request, id):
    camera = CameraConfig.objects.get(id=id)
    camera.delete()
    return redirect('camera_list')
@login_required(login_url='login')
def camera_update(request, id):
    camera = get_object_or_404(CameraConfig, id=id)
    print("Camera Type:", camera.camera_type) 

    if request.method == 'POST':
        camera.username = request.POST.get("username", camera.username)
        camera.password = request.POST.get("password", camera.password)
        camera.ip = request.POST.get("ip", camera.ip)
        camera.channel_name = request.POST.get("channel_name", camera.channel_name)

        check_option = request.POST.get("check_option") 
        valid_choices = dict(DETECTION_CHOICES).keys() 
        if check_option in valid_choices:
            camera.camera_type = check_option
        
        camera.save()
        return redirect('camera_list')

    context = {'camera': camera}
    return render(request, 'camera_update.html', context)

@login_required(login_url='login')
def detection_history(request):
    detectionRecords = detectionRecord.objects.all()
    context = {
        'detectionRecords': detectionRecords,
    }
    return render(request, 'detection_history.html', context)
@login_required(login_url='login')
def detection_delete(request, id):
    detection = detectionRecord.objects.get(id=id)
    detection.delete()
    messages.success(request, "Detection record Delete successfully.")
    return redirect('detection_history')
@login_required(login_url='login')
def detection_update(request, id):
    detection = get_object_or_404(detectionRecord, id=id)
    
    if request.method == 'POST':
        detection.vehicle_class = request.POST.get('vehicle_class', detection.vehicle_class)
        detection.plate_number = request.POST.get('plate_number', detection.plate_number)
        detection.has_helmet = request.POST.get('has_helmet') == 'on'
        detection.has_seatbelt = request.POST.get('has_seatbelt') == 'on'
        check_in_time = request.POST.get('check_in_time')
        check_out_time = request.POST.get('check_out_time')
        
        if check_in_time:
            detection.check_in_time = datetime.strptime(check_in_time, "%Y-%m-%dT%H:%M")

        if check_out_time:
            detection.check_out_time = datetime.strptime(check_out_time, "%Y-%m-%dT%H:%M")

        if 'vehicle_image' in request.FILES:
            detection.vehicle_image = request.FILES['vehicle_image']

        detection.save()
        messages.success(request, "Detection record updated successfully.")
        return redirect('detection_history') 

    return render(request, 'detection_update.html', {'detection': detection})
@login_required(login_url='login')
def view_images(request, id):
    detection = get_object_or_404(detectionRecord, id=id)
    images = {
        'vehicle_image': detection.vehicle_image.url if detection.vehicle_image else None,
        'license_plate_image': detection.license_plate_image.url if detection.license_plate_image else None,
        'helmet_image': detection.helmet_image.url if detection.helmet_image else None,
        'seatbelt_image': detection.seatbelt_image.url if detection.seatbelt_image else None,
        'full_frame_image': detection.full_frame_image.url if detection.full_frame_image else None,
    }
    context = {
        'detection': detection,
        'images': images
    }
    return render(request, 'image_gallery.html', context)

def stop():
    global global_stop_signal
    print("Not Set : ",global_stop_signal)
    global_stop_signal.set()
    print("Set : ",global_stop_signal)
def clear():
    global global_stop_signal
    global_stop_signal.clear()

def stop_detection(request):
    try:
        success, message = stop_all_camera_detections()
        return JsonResponse({"success": success, "message": message})
    except Exception as e:
        return JsonResponse({"success": False, "message": f"Error: {str(e)}"})

def start_detections_view(request):
    try:
        success, message = start_camera_detections(request)
        return JsonResponse({"success": success, "message": message})
    except Exception as e:
        return JsonResponse({"success": False, "message": f"Error: {str(e)}"})

