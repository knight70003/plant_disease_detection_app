from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.home, name='home'),
    path('features/', views.features, name='features'),
    path('analytics/', views.analytics, name='analytics'),
    path('contact/', views.contact, name='contact'),
    path('signup/', views.signup, name='signup'),
]
