from django.urls import path
from .views import predict_tumor

urlpatterns = [
    path("predict/", predict_tumor, name="predict_tumor"),
]
