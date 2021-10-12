from django.db import models

# Create your models here.
class Station(models.Model):
    name = models.CharField(max_length=50)
    city = models.CharField(max_length=50)
    def __str__(self):
        return f"{self.name} ({self.city})"
class CCTV(models.Model):
    name = models.CharField(max_length=50)
    location = models.CharField(max_length=50)
    station = models.ForeignKey(Station, on_delete=models.CASCADE)
    def __str__(self):
        return f"{self.name} (Location: {self.location}, Station: {self.station})"
class Crime(models.Model):
    classification = models.CharField(max_length=50)
    correct = models.BooleanField(default=False)
    validated = models.BooleanField(default=False)
    date = models.DateTimeField(auto_now_add=True, blank=True)
    image = models.CharField(max_length=100, default="webcam/ong.png")
    def __str__(self):
        return f"{self.classification} ({self.correct}) - {self.date}"