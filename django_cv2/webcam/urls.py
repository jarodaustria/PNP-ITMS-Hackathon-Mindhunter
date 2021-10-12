"""django_cv2 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static

app_name = "webcam"
urlpatterns = [
    path('index/', index, name="index"),
    path('pending_crimes', pending_crimes, name="pending_crimes"),
    path('crimes', crimes, name="crimes"),
    path('cctv1/', Home, name="cctv1"),
    path('crimes/update/true/<int:pk>/',update_crime_true,name='update_crime_true'),
    path('crimes/update/false/<int:pk>/',update_crime_false,name='update_crime_false'),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)