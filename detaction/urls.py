from django.urls import path
from . import views
import detaction.auth_view as auth_view
urlpatterns = [
    path('signup/', auth_view.signup, name='signup'),
    path('login/', auth_view.user_login, name='login'),
    path('logout/', auth_view.logout_view, name='logout'),
    path('home/', views.home, name='home'),
    path('home/<int:stream_id>/', views.home, name='start_live_stream'),
    path('start_detection', views.start_detections_view, name='start_dection'),
    path('stop_detection', views.stop_detection, name='stop_dection'),
    path('camera/', views.generate_camera_url, name='camera'),
    path('camera_list/', views.Camera_list, name='camera_list'),
    path('camera_update/<int:id>/', views.camera_update, name='camera_update'),
    path('camera_delete/<int:id>/', views.camera_delete, name='camera_delete'),
    path('detection_history/', views.detection_history, name='detection_history'),
    path('detection_update/<int:id>/', views.detection_update, name='detection_update'),
    path('detection_delete/<int:id>/', views.detection_delete, name='detection_delete'),
    path('images/<int:id>/', views.view_images, name='view_images'),
    path('update-id-card/', views.update_id_card_no, name='update_id_card_no'),
]