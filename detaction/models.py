from django.db import models
import os

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin, Group, Permission
from django.db import models

class CustomUserManager(BaseUserManager):
    def create_user(self, email, first_name, last_name, password=None):
        if not email:
            raise ValueError("Users must have an email address")
        
        email = self.normalize_email(email)
        user = self.model(email=email, first_name=first_name, last_name=last_name)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, first_name, last_name, password=None):
        user = self.create_user(email, first_name, last_name, password)
        user.is_admin = True
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user

class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)
    
    groups = models.ManyToManyField(Group, related_name="customuser_set", blank=True)
    user_permissions = models.ManyToManyField(Permission, related_name="customuser_permissions_set", blank=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    def __str__(self):
        return self.email
DETECTION_CHOICES = [
    ("0", "Check-out"),
    ("1", "Check-in"),
]


def helmet_detection_upload_path(instance, filename):
    return os.path.join('helmet_detections', filename)

def license_plate_upload_path(instance, filename):
    return os.path.join('license_plates', filename)
class detectionRecord(models.Model):
    vehicle_id = models.IntegerField(null=True)
    plate_number = models.CharField(max_length=20, null=True)
    id_card_no = models.CharField(max_length=20, null=True, blank=True)
    has_helmet = models.BooleanField(null=True)
    has_seatbelt = models.BooleanField(null=True)
    seatbelt_image = models.ImageField(upload_to='seatbelt_images/', null=True)
    license_plate_image = models.ImageField(upload_to=license_plate_upload_path, null=True)
    helmet_image = models.ImageField(upload_to=helmet_detection_upload_path, null=True)
    confidence = models.CharField(max_length=10, null=True)
    vehicle_class = models.CharField(max_length=20, null=True)
    vehicle_image = models.ImageField(upload_to='vehicle_images/', null=True)
    full_frame_image = models.ImageField(upload_to='full_frames/', blank=True, null=True)
    check_in_time = models.DateTimeField(auto_now_add=True)  
    check_out_time = models.DateTimeField(null=True)
    detection_type = models.CharField(max_length=1,choices=DETECTION_CHOICES,default='1')
    status = models.TextField(null=True, blank=True)
    missed = models.BooleanField(default=False)
    channel_id= models.IntegerField(max_length=10, null=True, blank=True)  
    def __str__(self):
        return f"License Plate {self.plate_number}"
class CameraConfig(models.Model):
    username = models.CharField(max_length=100)
    password = models.CharField(max_length=10)
    ip = models.CharField(max_length=30)
    channel_name = models.IntegerField(max_length=10)
    generated_url = models.CharField(max_length=255, null= True,blank=True)
    camera_type = models.CharField(max_length=1, choices=DETECTION_CHOICES)
    def save(self, *args, **kwargs):
        self.generated_url = f"rtsp://{self.username}:{self.password}@{self.ip}/cam/realmonitor?channel={self.channel_name}&subtype=1"
        super().save(*args, **kwargs)
    def __str__(self):
        return f"channel_name {self.generated_url} - Stream {self.channel_name}"