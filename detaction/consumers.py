from channels.generic.websocket import AsyncWebsocketConsumer
import json
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.mediastreams import VideoStreamTrack
from asgiref.sync import sync_to_async
from .models import detectionRecord,CameraConfig
from .models import detectionRecord
from django.core.serializers import serialize
import cv2
from av import VideoFrame
from channels.db import database_sync_to_async


class DetectionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add(
            "detection_group",
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            "detection_group",
            self.channel_name
        )

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            # Process the received data if necessary
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Error in receive: {e}")
            pass

    async def send_detection(self, event):
        try:
            data = event["data"] 
            if "license_plate_id" in data and data["license_plate_id"]:
                license_plate = await self.get_license_plate(data["license_plate_id"])
                if license_plate:
                    updated_data = {
                        "type": "Without Helmet" if not license_plate.has_helmet else "With Helmet",
                        "confidence": data.get("confidence"),
                        "license_plate_id": data["license_plate_id"],
                        "status": data.get("status", "Unknown"),
                        "detection_type": data.get("detection_type"),
                        "timestamp": license_plate.check_out_time.timestamp() if license_plate.check_out_time else license_plate.check_in_time.timestamp(),
                        "plate_number": license_plate.plate_number,
                        "vehicle_class": license_plate.vehicle_class,
                        "image_url": license_plate.license_plate_image.url if license_plate.license_plate_image else None,
                        "helmet": "Yes" if license_plate.has_helmet else "No" if license_plate.has_helmet is not None else "N/A",
                        "seatbelt": "Yes" if license_plate.has_seatbelt else "No" if license_plate.has_seatbelt is not None else "N/A",
                        "check_in_time": data.get("check_in_time"),
                        "check_out_time": data.get("check_out_time"),
                        'channel_id': data.get('channel_id'),
                    
                    }
                    print(f"vechicle send camera id: {data.get('channel_id')}")  
                    print(f"vechicle send detection type: {data.get('detection_type')}") 
                    data = updated_data
                    print(f"Updated data: {data}")
            await self.send(text_data=json.dumps(data))
        except Exception as e:
            print(f"Error in send_detection: {e}")
    @sync_to_async
    def get_license_plate(self, license_plate_id):
        try:
            return detectionRecord.objects.get(id=license_plate_id)
        except detectionRecord.DoesNotExist:
            return None
class CameraStreamTrack(VideoStreamTrack):
    def __init__(self, camera_url):
        super().__init__()
        self.camera_url = camera_url
        self.cap = None
        self._connect_camera()

    def _connect_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.camera_url}")
    async def recv(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame, attempting to reconnect...")
                self._connect_camera()
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Failed to read frame after reconnection")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            
            pts, time_base = await self.next_timestamp()
            video_frame.pts = pts
            video_frame.time_base = time_base     
            return video_frame
        except Exception as e:
            print(f"Error in recv: {e}")
            raise
    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

class CameraStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        try:
            camera_id = self.scope['url_route']['kwargs']['camera_id']
            camera = await database_sync_to_async(CameraConfig.objects.get)(id=camera_id)
            self.camera_url = camera.generated_url
            
            self.pc = RTCPeerConnection()
            self.track = CameraStreamTrack(self.camera_url)
            self.pc.addTrack(self.track)
            
            await self.accept()
            
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            await self.send(text_data=json.dumps({
                "sdp": offer.sdp,
                "type": offer.type
            }))
            
        except Exception as e:
            print(f"Error in connect: {e}")
            await self.close()

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            print(f"Received data: {data}")
            
            if data["type"] == "answer":
                answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                await self.pc.setRemoteDescription(answer)
            
            elif data["type"] == "candidate" and data["candidate"]:
                candidate_data = data["candidate"]
                candidate_str = candidate_data["candidate"]
                
             
                parts = candidate_str.split()
                foundation = parts[0].split(':')[1] 
                component = int(parts[1])
                protocol = parts[2]
                priority = int(parts[3])
                ip = parts[4]
                port = int(parts[5])
                type = parts[7]

                # Create RTCIceCandidate with the correct parameters
                candidate = RTCIceCandidate(
                    foundation=foundation,
                    component=component,
                    protocol=protocol,
                    priority=priority,
                    ip=ip,
                    port=port,
                    type=type,
                    sdpMid=candidate_data["sdpMid"],
                    sdpMLineIndex=candidate_data["sdpMLineIndex"],
                )
                
                print(f"Adding ICE candidate: {candidate}")
                await self.pc.addIceCandidate(candidate)
                
        except Exception as e:
            print(f"Error in receive: {str(e)}")
            import traceback
            traceback.print_exc()

    async def disconnect(self, close_code):
        if hasattr(self, 'track'):
            self.track.__del__()
        if hasattr(self, 'pc'):
            await self.pc.close()