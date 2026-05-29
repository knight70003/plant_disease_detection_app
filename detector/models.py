from django.db import models

# Create your models here.

from django.db import models

class Prediction(models.Model):
    image = models.ImageField(upload_to="uploads/")
    disease = models.CharField(max_length=100)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.disease