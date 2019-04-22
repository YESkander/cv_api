from django.db import models
from django.conf import settings

class Image(models.Model):
	author = models.CharField(max_length=200)
	name = models.CharField(max_length=200)
	text = models.TextField()
	imagefile = models.FileField(upload_to='images/', null=True, verbose_name="")
	
	def __str__(self):
		return self.name + ": " + str(self.imagefile)
# Create your models here.
