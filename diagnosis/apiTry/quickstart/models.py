from django.db import models

class Image(models.Model):
    url = models.CharField(max_length=100, blank=True, default='')
    class Meta:
        ordering = ['url']
