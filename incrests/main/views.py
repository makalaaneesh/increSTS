from django.shortcuts import render
from main.models import *
from django.http import HttpResponse, HttpResponseRedirect, HttpRequest, Http404,JsonResponse
import subprocess
import os

# Create your views here.

def home(request):
	return render(request,'main/layout/home.djt',{})


def uploadtextfile(request):
	if request.method == 'POST':
		p = Process()
		p.file = request.FILES['textfile']
		p.clean_file = request.FILES['cleantextfile']
		p.save()
		file_url = p.file.url
		clean_file_url = p.clean_file.url
		base_cmd = 'spark-submit --packages graphframes:graphframes:0.2.0-spark2.0-s_2.11 incrests_spark.py'
		cmd = base_cmd + " " + str(clean_file_url) + " " + str(file_url) + " " + str(p.id)
		args = cmd.split()
		stdout = open("output","wb")
		proc = subprocess.Popen(args,stdout=stdout,env=os.environ.copy())
		print proc.pid
		print cmd
		p.pid = proc.pid
		p.save()
		return HttpResponseRedirect('/incrests/status/')
	return render(request,'main/sites/upload.djt',{})


def status(request):
	process = Process.objects.all()
	response={}
	response['procs'] = process
	return render(request,'main/sites/status.djt',response)