from django.contrib import admin
from .models import detectionRecord,CameraConfig,CustomUser

# Register your models here.

admin.site.register(detectionRecord)
admin.site.register(CameraConfig)
admin.site.register(CustomUser)
