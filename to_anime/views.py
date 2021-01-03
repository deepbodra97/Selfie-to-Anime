from django.shortcuts import render, redirect

from .models import Photo
from .forms import PhotoForm

import os

def index(request):
	form = PhotoForm()
	photos = None # Photo.objects.all()
	anime = None
	if request.method == 'POST':
		filled_form = PhotoForm(request.POST, request.FILES)
		if filled_form.is_valid():
			photo = filled_form.save()
			print("photo.file.path", photo.file.path)
			anime = Photo.objects.get(file=photo.file.name)
			filename, file_extension = os.path.splitext(photo.file.path)
			return render(request, 'to_anime/photo_list.html', {'form': form, 'original': filename+'-o'+file_extension, 'anime': anime})
	return render(request, 'to_anime/photo_list.html', {'form': form, 'photos': photos})