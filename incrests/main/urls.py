from django.conf.urls import include, url
from main import views

urlpatterns = [
	url(r'^$',views.home),
	url(r'^uploadpaper/$',views.uploadtextfile),
	url(r'^status/$',views.status)
]