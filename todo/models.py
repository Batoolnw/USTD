from django.db import models

class Todo(models.Model):
    text = models.CharField(max_length=40)
    comment = models.CharField(max_length=40)
    imglnk = models.CharField(max_length=250, default="")
    happy = models.BooleanField(default=True)
    complete = models.BooleanField(default=False)
    faces = models.ImageField(upload_to='faces/', null=True, blank=True)


    def __str__(self):
        return self.text

