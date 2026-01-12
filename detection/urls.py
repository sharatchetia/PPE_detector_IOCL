from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),

    path('', views.home, name='home'),
    path('mode2/', views.mode2, name='mode2'),

    path('video_feed/', views.video_feed, name='video_feed'),
    path('get_violation_status/', views.get_violation_status, name='get_violation_status'),

    path('download_receipt/', views.download_receipt, name='download_receipt'),
    path('download_excel/', views.download_excel, name='download_excel'),

    path('violations/', views.view_violations, name='view_violations'),
]
