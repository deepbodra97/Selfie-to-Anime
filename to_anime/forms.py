from PIL import Image
from django import forms
from django.core.files import File
from .models import Photo

from .to_anime import *

import os

class PhotoForm(forms.ModelForm):
    x = forms.FloatField(widget=forms.HiddenInput())
    y = forms.FloatField(widget=forms.HiddenInput())
    width = forms.FloatField(widget=forms.HiddenInput())
    height = forms.FloatField(widget=forms.HiddenInput())

    class Meta:
        model = Photo
        fields = ('file', 'x', 'y', 'width', 'height', )

    def save(self):
        photo = super(PhotoForm, self).save()

        x = self.cleaned_data.get('x')
        y = self.cleaned_data.get('y')
        w = self.cleaned_data.get('width')
        h = self.cleaned_data.get('height')

        filename, file_extension = os.path.splitext(photo.file.path)

        image = Image.open(photo.file)
        # image.save(filename+'-o'+file_extension)

        cropped_image = image.crop((x, y, w+x, h+y))
        cropped_image.save(filename+'-o'+file_extension)
        anime_image = array_to_img(predict(cropped_image), scale=False)
        anime_image.save(photo.file.path)
        return photo