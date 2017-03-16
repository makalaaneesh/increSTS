from __future__ import unicode_literals

from django.db import models

# Create your models here.
def file_path(instance, filename):
	return '/'.join(['file','textfile',filename,])

class Process(models.Model):
    """
    Description: Process
    """
    file = models.FileField(upload_to=file_path)
    date = models.DateTimeField(auto_now=True)
    pid = models.CharField(max_length=100,null=True,blank=True)
    status = models.BooleanField(default=False)
    def __unicode__(self):
    	return str(self.id)
