# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.utils.translation import activate
from django.http import HttpResponse
from .models import Image
import numpy as np
import urllib.request
import json
import cv2
import os
 
# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))
 
# Minimum number of difference that have to be found
# to consider the recognition valid
MIN_SCORE = 35.0
METHOD_DETECTOR = 'ORB'  # 'SIFT'
LOWE_RATIO = 1.5

@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
	
	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = _grab_image(stream=request.FILES["image"])
			
		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)
 
			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
 
			# load the image and convert
			image = _grab_image(url=url)
					
		# convert the image to grayscale
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# initiate ORB/SIFT detector
		if METHOD_DETECTOR   == 'ORB':
			finder = cv2.ORB_create()
		elif METHOD_DETECTOR == 'SIFT':
			finder = cv2.xfeatures2d.SIFT_create()
		
		# find keypoints and decriptors of image
		kp_i = finder.detect(image, None)
		kp_i, des_i = finder.compute(image, kp_i)
				
		for i in Image.objects.all():
			# load the model from sqlite3
			image_m = np.asarray(bytearray(i.imagefile.read()), dtype="uint8")
			# decode byte array
			image_model = cv2.imdecode(image_m, cv2.IMREAD_COLOR)
			# convert the model to grayscale
			image_model = cv2.cvtColor(image_model, cv2.COLOR_BGR2GRAY)
			
			# find keypoints and descriptors of model
			kp_m = finder.detect(image_model, None)
			kp_m, des_m = finder.compute(image_model, kp_m)
			
			# create BFMatcher object
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			
			# match descriptors
			matches = bf.match(des_i,des_m)
			# sort them in the order of their distance
			matches = sorted(matches, key = lambda x:x.distance)
			
			# Apply ratio test
			good = []
			good_sum = 0.0
			for m in matches:
				if m.distance <= LOWE_RATIO * matches[0].distance:
					#print(matches[m+1].distance)
					#print(m.distance)
					good_sum+=m.distance
					good.append(m)
			
			score = good_sum/len(good)
			print(score)
			
			if score < MIN_SCORE:
				name = i.name
				data.update({"name": name})
				return JsonResponse(data)
 
	# return a JSON response
	#HttpResponse(image_data, content_type="image/j")
	return JsonResponse(data)
 
def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
 
	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.request.urlopen(url)
			data = resp.read()
 
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
 
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image